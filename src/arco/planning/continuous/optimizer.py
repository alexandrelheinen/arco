"""TrajectoryOptimizer: two-stage spatial trajectory optimizer."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import KDTree

from arco.mapping.occupancy import Occupancy

logger = logging.getLogger(__name__)


class TrajectoryOptimizer:
    """Two-stage spatial trajectory optimizer for continuous planning.

    Refines a reference path from a global planner by minimizing a composite
    cost that balances traversal time, proximity to the reference, and obstacle
    avoidance.  The trajectory is discretized in space (fixed spatial step Δx)
    rather than in time, keeping the problem independent of velocity.

    Two-stage solver strategy:

    - **Stage 1** (global) — :func:`scipy.optimize.differential_evolution`
      explores the solution space without convexity assumptions, producing a
      candidate solution in a viable region of the cost landscape.
    - **Stage 2** (local) — :func:`scipy.optimize.minimize` (L-BFGS-B by
      default) refines the Stage-1 solution efficiently.

    Both stages minimize the same composite cost::

        J = w_time * L² + w_deviation * Σ d(pᵢ, ref)² + w_collision * Σ c(pᵢ)

    where *L* is the total path length (proportional to traversal time at
    constant speed), *d(pᵢ, ref)* is the distance from point *pᵢ* to the
    nearest reference-path sample, and *c(pᵢ)* is the collision penalty
    derived from the KD-tree occupancy structure.

    The optimizer is model-agnostic.  Model-specific constraints are
    encapsulated in a ``feasibility`` callable that returns ``False`` for any
    state that violates joint limits, workspace bounds, or kinematic
    constraints.  Infeasible candidate points incur a large penalty during
    optimization and are removed from the final output.

    Args:
        occupancy: Occupancy map used for collision penalty queries.  Must
            implement :meth:`~arco.mapping.Occupancy.nearest_obstacle`.
        feasibility: Callable that returns ``True`` when a candidate state
            satisfies all model-specific constraints.  Receives one position
            array of shape ``(D,)``.
        spatial_step: Spatial discretization step Δx (world units).  The
            reference path is resampled into evenly spaced points separated
            by this distance.
        weight_time: Weight on the squared path-length term.  Should dominate
            by default to drive the optimizer toward shorter trajectories.
        weight_deviation: Weight on the squared deviation-from-reference term.
        weight_collision: Weight on the collision penalty term.
        deviation_bound: Maximum per-axis deviation allowed for each
            intermediate point during optimization (world units).  Defines
            the per-variable bounds passed to both solvers.
        stage1_population_count: Population size multiplier for
            :func:`~scipy.optimize.differential_evolution`.
        stage1_max_iter: Maximum number of generations for Stage 1.
        stage2_method: Local solver method passed to
            :func:`~scipy.optimize.minimize`.  Defaults to ``"L-BFGS-B"``.
        stage2_max_iter: Maximum iterations for Stage 2.
        infeasibility_penalty: Large additive penalty applied to the cost
            for each infeasible candidate point.
    """

    def __init__(
        self,
        occupancy: Occupancy,
        feasibility: Callable[[np.ndarray], bool],
        spatial_step: float = 1.0,
        weight_time: float = 1.0,
        weight_deviation: float = 0.1,
        weight_collision: float = 10.0,
        deviation_bound: float = 2.0,
        stage1_population_count: int = 15,
        stage1_max_iter: int = 100,
        stage2_method: str = "L-BFGS-B",
        stage2_max_iter: int = 200,
        infeasibility_penalty: float = 1e6,
    ) -> None:
        """Initialize TrajectoryOptimizer.

        Args:
            occupancy: Occupancy map for collision penalty queries.
            feasibility: Callable that checks model-specific constraints.
            spatial_step: Spatial discretization step Δx (world units).
            weight_time: Weight on the squared path-length term.
            weight_deviation: Weight on squared deviation from reference.
            weight_collision: Weight on the collision penalty term.
            deviation_bound: Maximum per-axis deviation for intermediate
                points (world units).
            stage1_population_count: Population size for differential
                evolution.
            stage1_max_iter: Maximum generations for Stage 1.
            stage2_method: Local solver method for Stage 2.
            stage2_max_iter: Maximum iterations for Stage 2.
            infeasibility_penalty: Additive penalty per infeasible point.

        Raises:
            ValueError: If *spatial_step* or any weight is not positive.
        """
        if spatial_step <= 0:
            raise ValueError(
                f"spatial_step must be positive, got {spatial_step!r}."
            )
        if weight_time <= 0 or weight_deviation < 0 or weight_collision < 0:
            raise ValueError(
                "weight_time must be positive; weight_deviation and "
                "weight_collision must be non-negative."
            )
        if deviation_bound <= 0:
            raise ValueError(
                f"deviation_bound must be positive, got {deviation_bound!r}."
            )
        self._occupancy = occupancy
        self._feasibility = feasibility
        self._spatial_step = spatial_step
        self._w_time = weight_time
        self._w_deviation = weight_deviation
        self._w_collision = weight_collision
        self._deviation_bound = deviation_bound
        self._stage1_population_count = stage1_population_count
        self._stage1_max_iter = stage1_max_iter
        self._stage2_method = stage2_method
        self._stage2_max_iter = stage2_max_iter
        self._infeasibility_penalty = infeasibility_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        reference_path: Sequence[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        """Optimize a reference path into a time-efficient trajectory.

        Resamples the reference path at :attr:`spatial_step` intervals, then
        runs a two-stage optimization (differential evolution followed by a
        local refinement) on the intermediate points.  Start and goal are held
        fixed at the endpoints of the resampled path.

        Infeasible states in the final trajectory are removed before the result
        is returned.  If fewer than two feasible states remain, ``None`` is
        returned.

        Args:
            reference_path: Ordered sequence of waypoints output by a global
                planner.  Must contain at least two points.

        Returns:
            An ordered list of optimized state arrays, or ``None`` if the
            path cannot be optimized (fewer than two feasible states remain).

        Raises:
            ValueError: If *reference_path* contains fewer than two points.
        """
        if len(reference_path) < 2:
            raise ValueError(
                "reference_path must contain at least two points."
            )

        ref_pts = np.asarray(
            [np.asarray(p, dtype=float) for p in reference_path]
        )
        resampled = self._discretize(ref_pts)

        # If only two points, nothing to optimize.
        if len(resampled) <= 2:
            feasible = [p for p in resampled if self._feasibility(p)]
            return feasible if len(feasible) >= 2 else None

        intermediate_ref = resampled[1:-1]  # shape (M, D)
        dimension = intermediate_ref.shape[1]
        point_count = len(intermediate_ref)

        # Build per-variable bounds for the optimizer.
        opt_bounds = self._build_bounds(intermediate_ref, dimension)

        # Precompute KDTree on the resampled reference for deviation queries.
        ref_tree = KDTree(resampled)

        # Stage 1 — global exploration via differential evolution.
        logger.debug(
            "TrajectoryOptimizer: Stage 1 (differential_evolution), "
            "%d variables, popsize=%d, maxiter=%d",
            len(opt_bounds),
            self._stage1_population_count,
            self._stage1_max_iter,
        )
        stage1_result = differential_evolution(
            self._cost,
            bounds=opt_bounds,
            args=(resampled, ref_tree, dimension, point_count),
            maxiter=self._stage1_max_iter,
            popsize=self._stage1_population_count,
            seed=0,
            tol=1e-4,
            polish=False,
        )
        x1 = stage1_result.x
        logger.debug(
            "TrajectoryOptimizer: Stage 1 complete, cost=%.4f, " "success=%s",
            stage1_result.fun,
            stage1_result.success,
        )

        # Stage 2 — local refinement from Stage-1 solution.
        logger.debug(
            "TrajectoryOptimizer: Stage 2 (%s), maxiter=%d",
            self._stage2_method,
            self._stage2_max_iter,
        )
        stage2_result = minimize(
            self._cost,
            x1,
            args=(resampled, ref_tree, dimension, point_count),
            method=self._stage2_method,
            bounds=opt_bounds,
            options={"maxiter": self._stage2_max_iter},
        )
        x_final = stage2_result.x
        logger.debug(
            "TrajectoryOptimizer: Stage 2 complete, cost=%.4f, " "success=%s",
            stage2_result.fun,
            stage2_result.success,
        )

        # Reconstruct and filter infeasible points.
        final_points = self._decode(x_final, resampled, dimension, point_count)
        feasible = [p for p in final_points if self._feasibility(p)]

        if len(feasible) < 2:
            logger.debug(
                "TrajectoryOptimizer: fewer than 2 feasible points — "
                "returning None."
            )
            return None

        logger.debug(
            "TrajectoryOptimizer: %d feasible points in output "
            "(%d removed).",
            len(feasible),
            len(final_points) - len(feasible),
        )
        return feasible

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        x: np.ndarray,
        reference: np.ndarray,
        ref_tree: KDTree,
        dimension: int,
        point_count: int,
    ) -> float:
        """Evaluate the composite trajectory cost.

        Args:
            x: Flat array of intermediate point coordinates, shape
                ``(point_count * dimension,)``.
            reference: Full resampled reference array, shape ``(N, D)``.
                First and last rows are the fixed start and goal.
            ref_tree: KDTree built on *reference* for fast deviation queries.
            dimension: Spatial dimension *D*.
            point_count: Number of intermediate points *M* (= N − 2).

        Returns:
            Scalar cost value.
        """
        points = self._decode(x, reference, dimension, point_count)

        # ------ T²: squared total path length ---------------------------
        segments = np.diff(points, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        time_cost = path_length**2

        # ------ Σ d(pᵢ, ref)²: squared deviation from reference --------
        dists_to_ref, _ = ref_tree.query(points)
        deviation_cost = float(np.sum(dists_to_ref**2))

        # ------ Σ c(pᵢ): collision penalty from occupancy map ----------
        collision_cost = 0.0
        for p in points:
            dist, _ = self._occupancy.nearest_obstacle(p)
            penetration = max(0.0, self._occupancy.clearance - dist)
            collision_cost += penetration

        # ------ Feasibility penalty (hard constraint proxy) -------------
        infeasibility_cost = 0.0
        for p in points:
            if not self._feasibility(p):
                infeasibility_cost += self._infeasibility_penalty

        return (
            self._w_time * time_cost
            + self._w_deviation * deviation_cost
            + self._w_collision * collision_cost
            + infeasibility_cost
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretize(self, reference: np.ndarray) -> np.ndarray:
        """Resample *reference* at uniform arc-length intervals of Δx.

        Args:
            reference: Reference waypoints array of shape ``(K, D)``.

        Returns:
            Resampled point array of shape ``(N, D)`` where consecutive
            points are approximately :attr:`spatial_step` apart.  Always
            includes the original start and goal as the first and last
            points.
        """
        diffs = np.diff(reference, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = float(cumulative[-1])

        if total_length == 0.0:
            return reference[[0, -1]]

        point_count = max(
            2, int(np.ceil(total_length / self._spatial_step)) + 1
        )
        sample_lengths = np.linspace(0.0, total_length, point_count)

        resampled = np.empty((point_count, reference.shape[1]))
        for i, s in enumerate(sample_lengths):
            idx = int(np.searchsorted(cumulative, s, side="right")) - 1
            idx = min(idx, len(reference) - 2)
            seg_end = float(cumulative[idx + 1])
            seg_start = float(cumulative[idx])
            if seg_end > seg_start:
                t = (s - seg_start) / (seg_end - seg_start)
            else:
                t = 0.0
            resampled[i] = reference[idx] + t * diffs[idx]

        # Snap endpoints exactly to the original start and goal.
        resampled[0] = reference[0]
        resampled[-1] = reference[-1]
        return resampled

    def _build_bounds(
        self,
        intermediate_ref: np.ndarray,
        dimension: int,
    ) -> List[Tuple[float, float]]:
        """Build per-variable bounds for the optimizer.

        Each coordinate of each intermediate point is bounded to
        ``[ref_coord − deviation_bound, ref_coord + deviation_bound]``.

        Args:
            intermediate_ref: Reference intermediate points, shape ``(M, D)``.
            dimension: Spatial dimension *D*.

        Returns:
            List of ``(lower, upper)`` bound pairs, length ``M * D``.
        """
        bounds: List[Tuple[float, float]] = []
        for p in intermediate_ref:
            for d in range(dimension):
                bounds.append(
                    (
                        p[d] - self._deviation_bound,
                        p[d] + self._deviation_bound,
                    )
                )
        return bounds

    def _decode(
        self,
        x: np.ndarray,
        reference: np.ndarray,
        dimension: int,
        point_count: int,
    ) -> np.ndarray:
        """Reconstruct the full point array from optimizer variables.

        Inserts the fixed start and goal from *reference* around the
        intermediate points encoded in *x*.

        Args:
            x: Flat variable array of length ``point_count * dimension``.
            reference: Full resampled reference array, shape ``(N, D)``.
            dimension: Spatial dimension *D*.
            point_count: Number of intermediate points *M*.

        Returns:
            Full trajectory array of shape ``(M + 2, D)``.
        """
        intermediate = x.reshape(point_count, dimension)
        return np.vstack([reference[0:1], intermediate, reference[-1:]])
