"""PlanningPipeline: algorithm-agnostic orchestrator for the planning pipeline.

The pipeline connects three injected stages — planner, pruner, optimizer —
by passing the output of each stage as the input to the next.  It is
deliberately independent of the specific algorithms inside each stage; it
only knows about their shared input/output contracts.

Pipeline stages
---------------
::

    start, goal
        │
        ▼
    ┌─────────┐
    │ Planner │  plan(start, goal) → list[np.ndarray] | None
    └────┬────┘
         │ raw_path
         ▼
    ┌─────────┐
    │ Pruner  │  prune(raw_path) → list[np.ndarray]   (optional)
    └────┬────┘
         │ pruned_path
         ▼
    ┌───────────┐
    │ Optimizer │  optimize(pruned_path) → TrajectoryResult   (optional)
    └───────────┘

Results from all stages are collected in a :class:`PipelineResult`
dataclass that can be serialised with :meth:`PlanningPipeline.save_result`
and reloaded with :meth:`PlanningPipeline.load_result`, enabling caching
between runs without re-running the full pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from arco.planning.continuous.base import ContinuousPlanner
    from arco.planning.continuous.optimizer import TrajectoryOptimizer
    from arco.planning.continuous.pruner import TrajectoryPruner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Snapshot of every stage output from one :meth:`PlanningPipeline.run` call.

    Stores the outputs from all stages so that callers can inspect or replay
    any intermediate result without re-running the pipeline.

    Attributes:
        raw_path: Unprocessed path returned by the planner, or ``None`` if
            planning failed.
        pruned_path: Path after pruning (fewer waypoints).  ``None`` when
            pruning was skipped or planning failed.
        trajectory: Time-parameterised trajectory states from the optimizer.
            ``None`` when optimization was skipped or failed.
        durations: Per-segment durations from the optimizer (seconds).
            ``None`` when optimization was skipped.
        total_duration: Sum of per-segment durations (seconds).
        planner_time: Wall-clock time spent in the planner (seconds).
        pruner_time: Wall-clock time spent in the pruner (seconds).
        optimizer_time: Wall-clock time spent in the optimizer (seconds).
        planner_status: ``'success'``, ``'no_path'``, or ``'not_run'``.
        optimizer_status: Optimizer status text, ``'skipped'``, or
            ``'not_run'``.
        optimizer_success: ``True`` when the optimizer converged.
        extra: Arbitrary extra metadata (e.g. node counts, costs).
    """

    raw_path: Optional[list[np.ndarray]] = None
    pruned_path: Optional[list[np.ndarray]] = None
    trajectory: Optional[list[np.ndarray]] = None
    durations: Optional[list[float]] = None
    total_duration: float = 0.0
    planner_time: float = 0.0
    pruner_time: float = 0.0
    optimizer_time: float = 0.0
    planner_status: str = "not_run"
    optimizer_status: str = "not_run"
    optimizer_success: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline manager
# ---------------------------------------------------------------------------


class PlanningPipeline:
    """Algorithm-agnostic orchestrator for the full planning pipeline.

    Connects three injected stages (planner, pruner, optimizer) by passing the
    output of each stage as the input to the next.  The pipeline is independent
    of the specific algorithms; it only knows about their shared contracts.

    Example usage::

        pipeline = PlanningPipeline(
            planner=RRTPlanner(occ, bounds=bounds, step_size=step_size),
            pruner=TrajectoryPruner(occ, step_size=step_size),
            optimizer=TrajectoryOptimizer(occ, cruise_speed=1.0),
        )
        result = pipeline.run(start, goal)
        if result.trajectory:
            PlanningPipeline.save_result(result, "cache/rrt.npz")

    Args:
        planner: Any :class:`~arco.planning.continuous.base.ContinuousPlanner`
            (or duck-typed equivalent) exposing
            ``plan(start, goal) → list[np.ndarray] | None``.
        pruner: Optional :class:`~arco.planning.continuous.pruner.TrajectoryPruner`.
            When ``None`` the raw path is forwarded directly to the optimizer.
        optimizer: Optional :class:`~arco.planning.continuous.optimizer.TrajectoryOptimizer`.
            When ``None`` the (pruned) path is stored directly as the trajectory
            with unit durations (``1.0 s`` per segment).
    """

    def __init__(
        self,
        planner: Optional["ContinuousPlanner"] = None,
        pruner: Optional["TrajectoryPruner"] = None,
        optimizer: Optional["TrajectoryOptimizer"] = None,
    ) -> None:
        """Initialize the pipeline with its stage objects.

        Args:
            planner: Optional planner instance — must expose
                ``plan(start, goal)``.  Required when calling :meth:`run`;
                may be ``None`` when only :meth:`run_from_path` is used.
            pruner: Optional pruner instance — must expose ``prune(path)``.
            optimizer: Optional optimizer — must expose ``optimize(path)``.
        """
        self.planner = planner
        self.pruner = pruner
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> "PipelineResult":
        """Run the full pipeline from planning through optimization.

        Executes each configured stage in order, passing the output of one
        stage as the input to the next.  Timing is recorded per stage.

        When planning fails (returns ``None``), the remaining stages are
        skipped and the returned :class:`PipelineResult` reflects the failure.

        The optional *progress* callback is called at the start of each stage::

            progress(stage_name: str, stage_index: int, total_stages: int)

        Args:
            start: Start configuration as a numpy array.
            goal: Goal configuration as a numpy array.
            progress: Optional callback for progress reporting.  Called with
                ``(stage_name, stage_index, total_stages)`` at the beginning
                of each stage.

        Returns:
            :class:`PipelineResult` with outputs and timing from every stage.
        """
        total = (
            1
            + (1 if self.pruner is not None else 0)
            + (1 if self.optimizer is not None else 0)
        )
        stage = 0
        result = PipelineResult()

        if self.planner is None:
            raise RuntimeError(
                "PlanningPipeline.run() requires a planner. "
                "Use run_from_path() when no planner is configured."
            )

        # ---- Stage 1: Planning ----------------------------------------
        stage += 1
        if progress is not None:
            progress("planning", stage, total)
        t0 = time.perf_counter()
        raw_path = self.planner.plan(start, goal)
        result.planner_time = time.perf_counter() - t0

        if raw_path is None or len(raw_path) < 2:
            result.planner_status = "no_path"
            logger.warning("PlanningPipeline: planner returned no path.")
            return result

        result.planner_status = "success"
        result.raw_path = raw_path
        active_path = raw_path

        # ---- Stage 2: Pruning (optional) --------------------------------
        if self.pruner is not None:
            stage += 1
            if progress is not None:
                progress("pruning", stage, total)
            t0 = time.perf_counter()
            pruned = self.pruner.prune(active_path)
            result.pruner_time = time.perf_counter() - t0
            result.pruned_path = pruned
            active_path = pruned
            logger.debug(
                "PlanningPipeline: pruned %d → %d nodes.",
                len(raw_path),
                len(pruned),
            )

        # ---- Stage 3: Optimization (optional) ---------------------------
        if self.optimizer is not None:
            stage += 1
            if progress is not None:
                progress("optimization", stage, total)
            t0 = time.perf_counter()
            opt_result = self.optimizer.optimize(active_path)
            result.optimizer_time = time.perf_counter() - t0
            result.optimizer_success = bool(opt_result.optimizer_success)
            result.optimizer_status = (
                f"{opt_result.optimizer_status_code}: "
                f"{opt_result.optimizer_status_text}"
            )
            if opt_result.states:
                result.trajectory = list(opt_result.states)
                result.durations = list(opt_result.durations)
                result.total_duration = float(sum(opt_result.durations))
            else:
                # Optimizer returned nothing — fall back to the pruned path.
                result.trajectory = list(active_path)
                n = max(len(active_path) - 1, 1)
                result.durations = [1.0] * n
                result.total_duration = float(n)
        else:
            # No optimizer: treat the path itself as the trajectory.
            result.trajectory = list(active_path)
            n = max(len(active_path) - 1, 1)
            result.durations = [1.0] * n
            result.total_duration = float(n)
            result.optimizer_status = "skipped"

        logger.info(
            "PlanningPipeline: done — planner %.2fs, pruner %.2fs, "
            "optimizer %.2fs; trajectory %d pts, %.1fs.",
            result.planner_time,
            result.pruner_time,
            result.optimizer_time,
            len(result.trajectory or []),
            result.total_duration,
        )
        return result

    def run_from_path(
        self,
        raw_path: list[np.ndarray],
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> "PipelineResult":
        """Run the pruner and optimizer stages on a pre-planned path.

        Skips the planner stage entirely.  Useful when a scene has already
        called its planner (e.g. to collect the search tree for
        visualization) and only needs the pruner + optimizer stages from
        the pipeline.

        When *raw_path* is empty or ``None``, the result reflects a
        ``'no_path'`` failure immediately.

        Args:
            raw_path: Pre-planned path as an ordered list of configuration
                arrays.  Must contain at least two points.
            progress: Optional callback ``(stage_name, stage_index, total)``
                invoked at the start of each stage.

        Returns:
            :class:`PipelineResult` with pruner/optimizer outputs and
            timing.  ``planner_status`` is set to ``'pre_planned'`` and
            ``planner_time`` is ``0.0``.
        """
        result = PipelineResult()
        result.planner_status = "pre_planned"

        if raw_path is None or len(raw_path) < 2:
            result.planner_status = "no_path"
            return result

        result.raw_path = list(raw_path)
        active_path: list[np.ndarray] = list(raw_path)

        total = (1 if self.pruner is not None else 0) + (
            1 if self.optimizer is not None else 0
        )
        stage = 0

        # ---- Stage: Pruning (optional) ----------------------------------
        if self.pruner is not None:
            stage += 1
            if progress is not None:
                progress("pruning", stage, total)
            t0 = time.perf_counter()
            pruned = self.pruner.prune(active_path)
            result.pruner_time = time.perf_counter() - t0
            result.pruned_path = pruned
            active_path = pruned
            logger.debug(
                "PlanningPipeline.run_from_path: pruned %d → %d nodes.",
                len(raw_path),
                len(pruned),
            )

        # ---- Stage: Optimization (optional) -----------------------------
        if self.optimizer is not None:
            stage += 1
            if progress is not None:
                progress("optimization", stage, total)
            t0 = time.perf_counter()
            opt_result = self.optimizer.optimize(active_path)
            result.optimizer_time = time.perf_counter() - t0
            result.optimizer_success = bool(opt_result.optimizer_success)
            result.optimizer_status = (
                f"{opt_result.optimizer_status_code}: "
                f"{opt_result.optimizer_status_text}"
            )
            if opt_result.states:
                result.trajectory = list(opt_result.states)
                result.durations = list(opt_result.durations)
                result.total_duration = float(sum(opt_result.durations))
            else:
                result.trajectory = list(active_path)
                n = max(len(active_path) - 1, 1)
                result.durations = [1.0] * n
                result.total_duration = float(n)
        else:
            result.trajectory = list(active_path)
            n = max(len(active_path) - 1, 1)
            result.durations = [1.0] * n
            result.total_duration = float(n)
            result.optimizer_status = "skipped"

        logger.info(
            "PlanningPipeline.run_from_path: pruner %.2fs, optimizer %.2fs;"
            " trajectory %d pts, %.1fs.",
            result.pruner_time,
            result.optimizer_time,
            len(result.trajectory or []),
            result.total_duration,
        )
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_result(result: "PipelineResult", path: str | Path) -> None:
        """Save a :class:`PipelineResult` to a compressed ``.npz`` file.

        Arrays (``raw_path``, ``pruned_path``, ``trajectory``) are stored
        as stacked numpy arrays.  Scalar and string metadata are stored as
        zero-dimensional numpy object arrays in the same file.

        Args:
            result: Pipeline result to serialise.
            path: Destination file path.  The ``.npz`` extension is added
                automatically if not already present.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, Any] = {}

        def _save_path(key: str, pts: list[np.ndarray] | None) -> None:
            if pts is not None and len(pts) > 0:
                arrays[key] = np.array([p for p in pts], dtype=float)
            else:
                arrays[f"{key}__empty"] = np.array([], dtype=float)

        _save_path("raw_path", result.raw_path)
        _save_path("pruned_path", result.pruned_path)
        _save_path("trajectory", result.trajectory)

        if result.durations is not None:
            arrays["durations"] = np.array(result.durations, dtype=float)

        # Metadata stored as JSON string inside the archive.
        meta = {
            "total_duration": result.total_duration,
            "planner_time": result.planner_time,
            "pruner_time": result.pruner_time,
            "optimizer_time": result.optimizer_time,
            "planner_status": result.planner_status,
            "optimizer_status": result.optimizer_status,
            "optimizer_success": result.optimizer_success,
            "extra": result.extra,
        }
        arrays["__meta__"] = np.array(json.dumps(meta))

        np.savez_compressed(str(path), **arrays)
        logger.debug("PlanningPipeline: saved result to %s.", path)

    @staticmethod
    def load_result(path: str | Path) -> "PipelineResult":
        """Load a :class:`PipelineResult` from a previously saved ``.npz`` file.

        Args:
            path: Path to the ``.npz`` file.  The ``.npz`` suffix is added
                automatically if not present.

        Returns:
            Reconstructed :class:`PipelineResult`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unrecognised.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")

        archive = np.load(str(path), allow_pickle=False)

        def _load_path(key: str) -> list[np.ndarray] | None:
            if key in archive:
                arr = archive[key]
                return [arr[i] for i in range(len(arr))]
            return None

        result = PipelineResult(
            raw_path=_load_path("raw_path"),
            pruned_path=_load_path("pruned_path"),
            trajectory=_load_path("trajectory"),
            durations=(
                list(archive["durations"].tolist())
                if "durations" in archive
                else None
            ),
        )
        if "__meta__" in archive:
            meta = json.loads(str(archive["__meta__"]))
            result.total_duration = float(meta.get("total_duration", 0.0))
            result.planner_time = float(meta.get("planner_time", 0.0))
            result.pruner_time = float(meta.get("pruner_time", 0.0))
            result.optimizer_time = float(meta.get("optimizer_time", 0.0))
            result.planner_status = str(meta.get("planner_status", "unknown"))
            result.optimizer_status = str(
                meta.get("optimizer_status", "unknown")
            )
            result.optimizer_success = bool(
                meta.get("optimizer_success", False)
            )
            result.extra = dict(meta.get("extra", {}))

        logger.debug("PlanningPipeline: loaded result from %s.", path)
        return result
