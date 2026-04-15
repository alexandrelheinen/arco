"""Vehicle benchmark dual-planner scene (RRT* vs SST) in 2-D."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from arco.config.palette import annotation_rgb, layer_rgb, obstacle_rgb, ui_rgb
from arco.tools.simulator import renderer_gl
from arco.tools.simulator.sim.tracking import VehicleConfig

_C_BG: tuple[int, int, int] = ui_rgb("background")

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)


def _polyline_length(path: list[np.ndarray] | None) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


class VehicleScene:
    """Shared-map RRT* vs SST benchmark scene for simulator race mode."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._planner = cfg.get("planner", cfg)
        self._world = cfg.get("world", {})
        self._vehicle = cfg.get("vehicle", {}).get("dubins", {})

        self._bounds = [tuple(b) for b in self._planner["bounds"]]
        self._start = np.array([2.0, 2.0], dtype=float)
        self._goal = np.array(
            [
                float(self._planner["bounds"][0][1]) - 2.0,
                float(self._planner["bounds"][1][1]) - 2.0,
            ],
            dtype=float,
        )

        self._occ: KDTreeOccupancy | None = None
        self._rrt_nodes: list[np.ndarray] = []
        self._rrt_parent: dict[int, int | None] = {}
        self._rrt_path: list[np.ndarray] | None = None
        self._rrt_traj: list[np.ndarray] = []

        self._sst_nodes: list[np.ndarray] = []
        self._sst_parent: dict[int, int | None] = {}
        self._sst_path: list[np.ndarray] | None = None
        self._sst_traj: list[np.ndarray] = []

        self._rrt_metrics: dict[str, Any] = {
            "steps": 0,
            "nodes": 0,
            "planner_time": 0.0,
            "planned_path_length": 0.0,
            "trajectory_arc_length": 0.0,
            "trajectory_duration": 0.0,
            "path_status": "stalled",
            "optimizer_status": "not-run",
        }
        self._sst_metrics: dict[str, Any] = dict(self._rrt_metrics)

    def build(self, *, progress=None) -> None:
        total = 4
        if progress is not None:
            progress("Building occupancy map", 1, total)
        self._occ = self._build_occupancy()

        if progress is not None:
            progress("Running RRT*", 2, total)
        rrt = RRTPlanner(
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._planner["rrt_max_sample_count"]),
            step_size=self._planner["step_size"],
            goal_tolerance=float(self._planner["goal_tolerance"]),
            collision_check_count=int(self._planner["collision_check_count"]),
            goal_bias=float(self._planner["goal_bias"]),
            early_stop=bool(self._planner.get("early_stop", True)),
        )
        t0 = time.perf_counter()
        self._rrt_nodes, self._rrt_parent, self._rrt_path = rrt.get_tree(
            self._start.copy(), self._goal.copy()
        )
        self._rrt_metrics.update(
            {
                "steps": (
                    max(0, len(self._rrt_path) - 1)
                    if self._rrt_path is not None
                    else 0
                ),
                "nodes": len(self._rrt_nodes),
                "planner_time": time.perf_counter() - t0,
                "planned_path_length": _polyline_length(self._rrt_path),
                "path_status": (
                    "found" if self._rrt_path is not None else "stalled"
                ),
            }
        )

        if progress is not None:
            progress("Running SST", 3, total)
        sst = SSTPlanner(
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._planner["sst_max_sample_count"]),
            step_size=self._planner["step_size"],
            goal_tolerance=float(self._planner["goal_tolerance"]),
            collision_check_count=int(self._planner["collision_check_count"]),
            goal_bias=float(self._planner["goal_bias"]),
            witness_radius=float(self._planner["witness_radius"]),
            early_stop=bool(self._planner.get("early_stop", True)),
        )
        t0 = time.perf_counter()
        self._sst_nodes, self._sst_parent, self._sst_path = sst.get_tree(
            self._start.copy(), self._goal.copy()
        )
        self._sst_metrics.update(
            {
                "steps": (
                    max(0, len(self._sst_path) - 1)
                    if self._sst_path is not None
                    else 0
                ),
                "nodes": len(self._sst_nodes),
                "planner_time": time.perf_counter() - t0,
                "planned_path_length": _polyline_length(self._sst_path),
                "path_status": (
                    "found" if self._sst_path is not None else "stalled"
                ),
            }
        )

        if progress is not None:
            progress("Optimizing trajectories", 4, total)
        (
            self._rrt_traj,
            self._rrt_metrics["trajectory_duration"],
            self._rrt_metrics["optimizer_status"],
        ) = self._optimize(self._rrt_path)
        (
            self._sst_traj,
            self._sst_metrics["trajectory_duration"],
            self._sst_metrics["optimizer_status"],
        ) = self._optimize(self._sst_path)
        self._rrt_metrics["trajectory_arc_length"] = _polyline_length(
            self._rrt_traj
        )
        self._sst_metrics["trajectory_arc_length"] = _polyline_length(
            self._sst_traj
        )

    def _optimize(
        self, path: list[np.ndarray] | None
    ) -> tuple[list[np.ndarray], float, str]:
        if path is None or len(path) < 2 or self._occ is None:
            return [], 0.0, "no-path"
        if bool(self._planner.get("enable_pruning", False)):
            pruner = TrajectoryPruner(
                self._occ,
                step_size=np.asarray(self._planner["step_size"], dtype=float),
                collision_check_count=int(
                    self._planner["collision_check_count"]
                ),
            )
            path = pruner.prune(path)
        else:
            path = list(path)
        try:
            opt = TrajectoryOptimizer(
                self._occ,
                cruise_speed=float(self._vehicle.get("cruise_speed", 3.0)),
                weight_time=10.0,
                weight_deviation=1.0,
                weight_velocity=1.0,
                weight_collision=20.0,
                sample_count=10,
                max_iter=150,
            )
            res = opt.optimize(path)
            return (
                list(res.states),
                float(sum(res.durations)),
                (f"{res.optimizer_status_code}: {res.optimizer_status_text}"),
            )
        except Exception:
            return [], 0.0, "exception"

    def _build_occupancy(self) -> KDTreeOccupancy:
        rng = np.random.default_rng(int(self._world.get("random_seed", 7)))
        x_max = float(self._planner["bounds"][0][1])
        y_max = float(self._planner["bounds"][1][1])

        spacing = float(self._world.get("wall_spacing", 1.5))
        gap_a = float(self._world.get("wall_gap_start_fraction", 0.60))
        gap_b = float(self._world.get("wall_gap_end_fraction", 0.70))
        wall_pts = [
            [x, y_max / 2.0] for x in np.arange(0.0, x_max * gap_a, spacing)
        ] + [
            [x, y_max / 2.0] for x in np.arange(x_max * gap_b, x_max, spacing)
        ]

        margin = float(self._world.get("corner_margin", 5.0))
        scatter_count = int(self._world.get("scatter_count", 40))
        scatter_pts: list[list[float]] = []
        while len(scatter_pts) < scatter_count:
            p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
            scatter_pts.append(p.tolist())

        return KDTreeOccupancy(
            wall_pts + scatter_pts,
            clearance=float(self._planner["obstacle_clearance"]),
        )

    @property
    def title(self) -> str:
        return "RRT* vs SST - vehicle benchmark"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        return _C_BG

    @property
    def world_points(self) -> list[tuple[float, float]]:
        return [
            (self._bounds[0][0], self._bounds[1][0]),
            (self._bounds[0][1], self._bounds[1][1]),
        ]

    @property
    def vehicle_config(self) -> VehicleConfig:
        return VehicleConfig(
            max_speed=float(self._vehicle.get("max_speed", 5.0)),
            min_speed=0.0,
            cruise_speed=float(self._vehicle.get("cruise_speed", 3.0)),
            lookahead_distance=float(self._vehicle.get("lookahead", 4.0)),
            goal_radius=float(self._vehicle.get("goal_radius", 3.0)),
            max_turn_rate=math.radians(
                float(self._vehicle.get("max_turn_rate", 90.0))
            ),
            max_acceleration=float(self._vehicle.get("max_accel", 4.9)),
            max_turn_rate_dot=math.radians(
                float(self._vehicle.get("max_turn_rate_dot", 3600.0))
            ),
            curvature_gain=float(self._vehicle.get("curvature_gain", 0.5)),
        )

    @property
    def rrt_total(self) -> int:
        return len(self._rrt_nodes)

    @property
    def sst_total(self) -> int:
        return len(self._sst_nodes)

    @property
    def rrt_waypoints(self) -> list[tuple[float, float]]:
        pts = self._rrt_traj if self._rrt_traj else self._rrt_path
        if pts is None:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    @property
    def sst_waypoints(self) -> list[tuple[float, float]]:
        pts = self._sst_traj if self._sst_traj else self._sst_path
        if pts is None:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        return dict(self._rrt_metrics)

    @property
    def sst_metrics(self) -> dict[str, Any]:
        return dict(self._sst_metrics)

    def draw_background(
        self, rrt_revealed: int, sst_revealed: int, racing: bool = False
    ) -> None:
        assert self._occ is not None
        _ob = obstacle_rgb()
        renderer_gl.draw_obstacle_points(
            self._occ.points,
            _ob[0] / 255.0,
            _ob[1] / 255.0,
            _ob[2] / 255.0,
            point_size=5.0,
        )

        if not racing:
            _re = layer_rgb("rrt", "tree")
            renderer_gl.draw_tree(
                self._rrt_nodes,
                self._rrt_parent,
                rrt_revealed,
                _re[0] / 255.0,
                _re[1] / 255.0,
                _re[2] / 255.0,
                _re[0] / 255.0,
                _re[1] / 255.0,
                _re[2] / 255.0,
            )
            _se = layer_rgb("sst", "tree")
            renderer_gl.draw_tree(
                self._sst_nodes,
                self._sst_parent,
                sst_revealed,
                _se[0] / 255.0,
                _se[1] / 255.0,
                _se[2] / 255.0,
                _se[0] / 255.0,
                _se[1] / 255.0,
                _se[2] / 255.0,
            )

            _rp = layer_rgb("rrt", "path")
            _rt = layer_rgb("rrt", "trajectory")
            if rrt_revealed >= self.rrt_total and self._rrt_path is not None:
                renderer_gl.draw_path(
                    self._rrt_path,
                    _rp[0] / 255.0,
                    _rp[1] / 255.0,
                    _rp[2] / 255.0,
                    width=1.6,
                    alpha=(0.35 if self._rrt_traj else 1.0),
                )
                if self._rrt_traj:
                    renderer_gl.draw_path(
                        self._rrt_traj,
                        _rt[0] / 255.0,
                        _rt[1] / 255.0,
                        _rt[2] / 255.0,
                        width=3.0,
                    )
            _sp = layer_rgb("sst", "path")
            _st = layer_rgb("sst", "trajectory")
            if sst_revealed >= self.sst_total and self._sst_path is not None:
                renderer_gl.draw_path(
                    self._sst_path,
                    _sp[0] / 255.0,
                    _sp[1] / 255.0,
                    _sp[2] / 255.0,
                    width=1.6,
                    alpha=(0.35 if self._sst_traj else 1.0),
                )
                if self._sst_traj:
                    renderer_gl.draw_path(
                        self._sst_traj,
                        _st[0] / 255.0,
                        _st[1] / 255.0,
                        _st[2] / 255.0,
                        width=3.0,
                    )
        else:
            _rt = layer_rgb("rrt", "trajectory")
            _rp = layer_rgb("rrt", "path")
            if self._rrt_traj:
                renderer_gl.draw_path(
                    self._rrt_traj,
                    _rt[0] / 255.0,
                    _rt[1] / 255.0,
                    _rt[2] / 255.0,
                    width=3.0,
                )
            elif self._rrt_path is not None:
                renderer_gl.draw_path(
                    self._rrt_path,
                    _rp[0] / 255.0,
                    _rp[1] / 255.0,
                    _rp[2] / 255.0,
                    width=2.0,
                )
            _st = layer_rgb("sst", "trajectory")
            _sp = layer_rgb("sst", "path")
            if self._sst_traj:
                renderer_gl.draw_path(
                    self._sst_traj,
                    _st[0] / 255.0,
                    _st[1] / 255.0,
                    _st[2] / 255.0,
                    width=3.0,
                )
            elif self._sst_path is not None:
                renderer_gl.draw_path(
                    self._sst_path,
                    _sp[0] / 255.0,
                    _sp[1] / 255.0,
                    _sp[2] / 255.0,
                    width=2.0,
                )

        _ann = annotation_rgb(dark_bg=True)
        _bg = _C_BG
        renderer_gl.draw_ring(
            float(self._start[0]),
            float(self._start[1]),
            1.2,
            0.6,
            _ann[0] / 255.0,
            _ann[1] / 255.0,
            _ann[2] / 255.0,
        )
        renderer_gl.draw_disc(
            float(self._start[0]),
            float(self._start[1]),
            0.6,
            _bg[0] / 255.0,
            _bg[1] / 255.0,
            _bg[2] / 255.0,
        )
        renderer_gl.draw_ring(
            float(self._goal[0]),
            float(self._goal[1]),
            1.2,
            0.6,
            _ann[0] / 255.0,
            _ann[1] / 255.0,
            _ann[2] / 255.0,
        )
        renderer_gl.draw_disc(
            float(self._goal[0]),
            float(self._goal[1]),
            0.6,
            _bg[0] / 255.0,
            _bg[1] / 255.0,
            _bg[2] / 255.0,
        )
