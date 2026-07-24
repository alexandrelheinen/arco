"""ReferencePath: arc-length parameterization of planner waypoints."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class ReferencePath:
    """Arc-length parameterized reference path for contouring MPC.

    Provides position, tangent, curvature, and pose-projection queries
    over an ordered waypoint polyline.

    Attributes:
        total_length: Total path length in meters.
        waypoint_count: Number of input waypoints.
    """

    def __init__(self, waypoints: Sequence[tuple[float, float]]) -> None:
        """Build a reference path from ordered waypoints.

        Args:
            waypoints: Ordered ``(x, y)`` points.  At least two distinct
                points are required.

        Raises:
            ValueError: If fewer than two waypoints are provided or the
                total length is zero.
        """
        pts = np.asarray(waypoints, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"waypoints must have shape (N, 2), got {pts.shape}."
            )
        if pts.shape[0] < 2:
            raise ValueError("ReferencePath requires at least two waypoints.")

        deltas = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        # Collapse zero-length segments to keep cumulative arc length
        # strictly non-decreasing without NaN tangents.
        keep = seg_lengths > 1e-12
        if not np.any(keep):
            raise ValueError("ReferencePath total length must be positive.")
        pts = np.vstack([pts[0], pts[1:][keep]])
        deltas = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        if float(cumulative[-1]) <= 0.0:
            raise ValueError("ReferencePath total length must be positive.")

        self._points = pts
        self._seg_lengths = seg_lengths
        self._cumulative = cumulative
        self._total_length = float(cumulative[-1])
        tangents = deltas / seg_lengths[:, None]
        # Per-vertex tangents: copy segment tangent to the start vertex,
        # last vertex reuses the final segment tangent.
        vertex_tangents = np.vstack([tangents, tangents[-1]])
        self._tangents = vertex_tangents
        headings = np.arctan2(vertex_tangents[:, 1], vertex_tangents[:, 0])
        self._headings = headings
        self._curvatures = self._finite_difference_curvature(
            cumulative, headings
        )

    # Spread a sharp polyline turn over at most this arc length (m) when
    # forming κ for speed limiting.  Long collinear segments must not dilute
    # a 90° kink into κ≈0 (else ``v_curve = ω/|κ|`` never brakes).
    _CURVATURE_DS_CAP_M: float = 20.0
    # Minimum spreading length (m) for a non-trivial turn.  Optimizer stubs
    # and A* grid corners can be ~1 m; ``Δψ/ds`` then yields Dirac-like κ
    # (|κ|≳1) that makes long-horizon ``v_curve`` NLPs fail (city purple
    # racer stuck at speed 0 after ``solve_failed``).  Floor near half a
    # city road half-width keeps braking meaningful without killing IPOPT.
    _CURVATURE_DS_FLOOR_M: float = 8.0
    # Hard cap on |κ| (1/m) after preview — safety net for the NLP.
    _CURVATURE_ABS_MAX: float = 0.35
    # Backward horizon (m) over which an upcoming corner κ is visible so the
    # NMPC can decelerate before the kink (≈ cruise² / (2 a) at city soft
    # limits).
    _CURVATURE_PREVIEW_DS_M: float = 40.0

    @classmethod
    def _finite_difference_curvature(
        cls,
        cumulative: np.ndarray,
        headings: np.ndarray,
    ) -> np.ndarray:
        """Estimate curvature from consecutive segment heading changes.

        Each interior vertex uses the turn between its incoming and outgoing
        headings, spread over ``min(ds_in, ds_out, ds_cap)`` and, for
        non-trivial turns, at least ``ds_floor`` so short polyline stubs do
        not create Dirac κ.  A skip-one finite difference
        (``h[i+1] - h[i-1]``) can report ``κ ≈ 0`` on a 90° L-kink when
        those headings cancel, which disabled curve-speed limiting on city
        A*/RRT* polylines.  A backward max-preview then keeps the corner
        ``κ`` visible on the approach so braking starts before the vertex.
        """
        kappa = np.zeros_like(headings)
        n = len(headings)
        ds_cap = float(cls._CURVATURE_DS_CAP_M)
        ds_floor = float(cls._CURVATURE_DS_FLOOR_M)
        for i in range(1, n - 1):
            ds_in = float(cumulative[i] - cumulative[i - 1])
            ds_out = float(cumulative[i + 1] - cumulative[i])
            ds = min(ds_in, ds_out, ds_cap)
            if ds < 1e-12:
                continue
            dtheta = math.atan2(
                math.sin(headings[i] - headings[i - 1]),
                math.cos(headings[i] - headings[i - 1]),
            )
            if abs(dtheta) > 1e-6:
                ds = max(ds, ds_floor)
            kappa[i] = dtheta / ds
        if n >= 2:
            kappa[0] = kappa[1]
            kappa[-1] = kappa[-2]

        preview = float(cls._CURVATURE_PREVIEW_DS_M)
        if preview <= 0.0 or n < 2:
            previewed = kappa
        else:
            previewed = kappa.copy()
            for i in range(n):
                s_i = float(cumulative[i])
                best = float(previewed[i])
                best_abs = abs(best)
                for j in range(i + 1, n):
                    if float(cumulative[j]) - s_i > preview:
                        break
                    cand = float(kappa[j])
                    if abs(cand) > best_abs:
                        best = cand
                        best_abs = abs(cand)
                previewed[i] = best

        k_max = float(cls._CURVATURE_ABS_MAX)
        if k_max > 0.0:
            previewed = np.clip(previewed, -k_max, k_max)
        return previewed

    @property
    def total_length(self) -> float:
        """Total arc length of the path (m)."""
        return self._total_length

    @property
    def waypoint_count(self) -> int:
        """Number of waypoints after zero-length filtering."""
        return int(self._points.shape[0])

    def _clamp_s(self, s: float) -> float:
        return float(np.clip(s, 0.0, self._total_length))

    def _segment_index(self, s: float) -> int:
        s = self._clamp_s(s)
        idx = int(np.searchsorted(self._cumulative, s, side="right") - 1)
        return int(np.clip(idx, 0, len(self._seg_lengths) - 1))

    def position(self, s: float) -> tuple[float, float]:
        """Return the reference position at arc length *s*.

        Args:
            s: Arc length along the path (m).

        Returns:
            ``(x_ref, y_ref)`` in world frame.
        """
        s = self._clamp_s(s)
        i = self._segment_index(s)
        ds = s - float(self._cumulative[i])
        length = float(self._seg_lengths[i])
        alpha = 0.0 if length < 1e-12 else ds / length
        p0 = self._points[i]
        p1 = self._points[i + 1]
        pos = (1.0 - alpha) * p0 + alpha * p1
        return float(pos[0]), float(pos[1])

    def tangent(self, s: float) -> tuple[float, float]:
        """Return the unit tangent at arc length *s*.

        Args:
            s: Arc length along the path (m).

        Returns:
            ``(cos θ_ref, sin θ_ref)``.
        """
        s = self._clamp_s(s)
        i = self._segment_index(s)
        t = self._tangents[i]
        return float(t[0]), float(t[1])

    def heading(self, s: float) -> float:
        """Return the reference heading at arc length *s*.

        Args:
            s: Arc length along the path (m).

        Returns:
            Heading in radians.
        """
        tx, ty = self.tangent(s)
        return math.atan2(ty, tx)

    def curvature(self, s: float) -> float:
        """Return an approximate curvature at arc length *s*.

        Args:
            s: Arc length along the path (m).

        Returns:
            Curvature in 1/m (finite-difference estimate).
        """
        s = self._clamp_s(s)
        return float(np.interp(s, self._cumulative, self._curvatures))

    def sample(self, sample_count: int) -> tuple[np.ndarray, ...]:
        """Sample path quantities on a uniform arc-length grid.

        Args:
            sample_count: Number of samples (at least 2).

        Returns:
            Tuple ``(s, x, y, heading, curvature)`` each of shape
            ``(sample_count,)``.
        """
        count = max(int(sample_count), 2)
        s_vals = np.linspace(0.0, self._total_length, count)
        xs = np.empty(count)
        ys = np.empty(count)
        headings = np.empty(count)
        kappas = np.empty(count)
        for i, s in enumerate(s_vals):
            xs[i], ys[i] = self.position(float(s))
            headings[i] = self.heading(float(s))
            kappas[i] = self.curvature(float(s))
        return s_vals, xs, ys, headings, kappas

    def project(
        self,
        pose: tuple[float, float, float],
        *,
        s_hint: float | None = None,
        window: float | None = None,
    ) -> tuple[float, float, float]:
        """Project a pose onto the nearest path point.

        Args:
            pose: Vehicle pose ``(x, y, heading)``.
            s_hint: Optional arc-length center for a local search window.
                Used with *window* so contouring progress cannot flip to a
                distant junction-scale nearest segment after a corner cut.
            window: Half-width (m) of the local search around *s_hint*.
                Ignored unless *s_hint* is also provided.  ``None`` or a
                non-positive value keeps the global nearest-point search.

        Returns:
            ``(s, lateral_error, heading_error)`` where *lateral_error*
            is signed (left positive) and *heading_error* is wrapped to
            ``(−π, π]``.
        """
        x, y, theta = pose
        best_s = 0.0
        best_dist_sq = float("inf")
        best_lat = 0.0
        best_heading = 0.0

        s_lo = 0.0
        s_hi = self._total_length
        if s_hint is not None and window is not None and window > 0.0:
            s_lo = max(0.0, float(s_hint) - float(window))
            s_hi = min(self._total_length, float(s_hint) + float(window))

        for i, length in enumerate(self._seg_lengths):
            seg_start = float(self._cumulative[i])
            seg_end = float(self._cumulative[i + 1])
            if seg_end < s_lo or seg_start > s_hi:
                continue
            p0 = self._points[i]
            p1 = self._points[i + 1]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            if length < 1e-12:
                continue
            tx = dx / length
            ty = dy / length
            rel_x = x - float(p0[0])
            rel_y = y - float(p0[1])
            proj = rel_x * tx + rel_y * ty
            proj = float(np.clip(proj, 0.0, length))
            # Clamp the candidate into the local arc-length window.
            cand_s = float(np.clip(seg_start + proj, s_lo, s_hi))
            proj = cand_s - seg_start
            qx = float(p0[0]) + proj * tx
            qy = float(p0[1]) + proj * ty
            dist_sq = (x - qx) ** 2 + (y - qy) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_s = cand_s
                # Signed lateral: left of tangent is positive.
                best_lat = -(x - qx) * ty + (y - qy) * tx
                theta_ref = math.atan2(ty, tx)
                best_heading = math.atan2(
                    math.sin(theta - theta_ref),
                    math.cos(theta - theta_ref),
                )

        return best_s, float(best_lat), float(best_heading)
