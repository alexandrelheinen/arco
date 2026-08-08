"""Tests for arco.protocols structural contracts."""

from __future__ import annotations

import numpy as np

from arco.control.pure_pursuit import PurePursuitController
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping import EuclideanGrid, KDTreeOccupancy, ManhattanGrid
from arco.planning import AStarPlanner, RRTPlanner, TrajectoryPruner
from arco.planning.continuous.optimizer import TrajectoryOptimizer
from arco.protocols import (
    DiscreteMap,
    OccupancyLike,
    OptimizerLike,
    PathTracker,
    PlannerLike,
    PrunerLike,
    VehicleModel,
)


def test_grids_satisfy_discrete_map_protocol():
    assert isinstance(ManhattanGrid((3, 3)), DiscreteMap)
    assert isinstance(EuclideanGrid((3, 3)), DiscreteMap)


def test_kdtree_occupancy_satisfies_occupancy_like():
    occ = KDTreeOccupancy(np.array([[0.0, 0.0], [1.0, 1.0]]), clearance=0.5)
    assert isinstance(occ, OccupancyLike)


def test_rrt_satisfies_planner_like():
    occ = KDTreeOccupancy(np.array([[10.0, 10.0]]), clearance=0.1)
    planner = RRTPlanner(
        occ, bounds=[(0.0, 1.0), (0.0, 1.0)], max_sample_count=1
    )
    assert isinstance(planner, PlannerLike)


def test_pruner_and_optimizer_protocols():
    occ = KDTreeOccupancy(np.array([[10.0, 10.0]]), clearance=0.1)
    pruner = TrajectoryPruner(occ, step_size=np.array([1.0, 1.0]))
    opt = TrajectoryOptimizer(occ, weight_time=1.0)
    assert isinstance(pruner, PrunerLike)
    assert isinstance(opt, OptimizerLike)


def test_pure_pursuit_satisfies_path_tracker():
    assert isinstance(PurePursuitController(), PathTracker)


def test_dubins_vehicle_satisfies_vehicle_model():
    assert isinstance(DubinsVehicle(), VehicleModel)


def test_astar_graph_satisfies_discrete_map():
    grid = ManhattanGrid((4, 4))
    planner = AStarPlanner(grid)
    assert isinstance(planner.graph, DiscreteMap)
