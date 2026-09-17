"""What the compiled boundary refuses, and how it says so.

Every exported function returns a ``PyResult``, so a rejected argument
has to arrive in Python as an ordinary exception rather than as a panic
crossing the boundary. ``FR-SAFE-01`` and ADR-009 require that, and an
unwind converted by PyO3 derives from ``BaseException``, which an
``except Exception`` handler passes over. Each test below therefore
pins the exception type and its message, not merely that something was
raised.

The guidance section pairs each rejection with the value the same
attribute accepts, because a setter that refuses everything would pass a
rejection test on its own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco._arco import (
    AStar,
    AStarPlanner,
    BSplineInterpolator,
    ContinuousPlanner,
    DiscretePlanner,
    DubinsPrimitive,
    DubinsVehicle,
    ExplorationPrimitive,
    Interpolator,
    MovingAverageInterpolator,
    PlannerCost,
    RouteRouter,
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
    TrajectoryResult,
)
from arco.mapping import (
    CartesianGraph,
    EuclideanGrid,
    KDTreeOccupancy,
    ManhattanGrid,
)
from arco.planning.continuous.telemetry import PlannerTelemetry

# Comparison tolerances for the assertions below, in meters and radians.
METERS = 1e-12
RADIANS = 1e-12

BOUNDS_2D = [(0.0, 10.0), (0.0, 10.0)]


def free_space():
    """An occupancy whose only obstacle sits far outside the bounds."""
    return KDTreeOccupancy([[50.0, 50.0]], clearance=0.3)


def straight_graph():
    """Three nodes in a row, one meter apart, joined by two edges."""
    graph = CartesianGraph()
    for node, x in enumerate([0.0, 1.0, 2.0]):
        graph.add_node(node, x, 0.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return graph


# ---------------------------------------------------------------------
# The abstract bases raise rather than answering with a default
# ---------------------------------------------------------------------


def test_the_base_interpolator_declines_to_smooth_a_path():
    with pytest.raises(
        NotImplementedError,
        match="Interpolator.interpolate is not implemented",
    ):
        Interpolator().interpolate([[0.0, 0.0], [1.0, 1.0]])


def test_the_base_primitive_declines_to_produce_a_segment():
    with pytest.raises(
        NotImplementedError,
        match="ExplorationPrimitive.steer is not implemented",
    ):
        ExplorationPrimitive().steer([0.0, 0.0], [1.0, 1.0])


def test_the_base_continuous_planner_declines_to_plan():
    with pytest.raises(
        NotImplementedError,
        match=r"ContinuousPlanner\.plan is abstract\.",
    ):
        ContinuousPlanner(free_space()).plan([0.0, 0.0], [1.0, 1.0])


def test_the_default_cost_measures_the_straight_line():
    cost = PlannerCost()
    assert cost.distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        5.0, abs=METERS
    )
    assert cost.heuristic([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        5.0, abs=METERS
    )


def test_the_default_cost_rejects_two_states_of_different_length():
    with pytest.raises(ValueError, match="point has dimension 1, expected 2"):
        PlannerCost().distance([0.0, 0.0], [1.0])


# ---------------------------------------------------------------------
# Interpolators
# ---------------------------------------------------------------------


def test_a_spline_of_degree_zero_is_a_step_and_is_refused():
    with pytest.raises(
        ValueError, match="spline degree is 0, which is outside"
    ):
        BSplineInterpolator(0)


def test_the_spline_degree_can_be_raised_but_not_dropped_to_zero():
    interpolator = BSplineInterpolator(3)
    interpolator.degree = 5
    assert interpolator.degree == 5

    with pytest.raises(
        ValueError, match="spline degree is 0, which is outside"
    ):
        interpolator.degree = 0
    assert interpolator.degree == 5


def test_the_spline_hands_back_the_path_it_was_given():
    """Deviation C-12: the Python class stored a degree and returned
    its argument, and the port carries that across unchanged."""
    path = [[0.0, 0.0], [1.0, 5.0], [2.0, 0.0]]
    smoothed = BSplineInterpolator(3).interpolate(path)
    assert [list(point) for point in smoothed] == path


def test_a_smoothing_pass_count_below_one_is_refused():
    with pytest.raises(
        ValueError, match="smoothing passes is 0, which is outside"
    ):
        MovingAverageInterpolator(iterations=0)


@pytest.mark.parametrize("window", [0, 2, 4])
def test_a_smoothing_window_that_is_not_an_odd_count_is_refused(window):
    with pytest.raises(
        ValueError, match="smoothing window is .*, which is outside"
    ):
        MovingAverageInterpolator(window=window)


def test_the_smoothing_settings_revalidate_when_they_are_replaced():
    interpolator = MovingAverageInterpolator(iterations=1, window=3)
    interpolator.iterations = 4
    interpolator.window = 5
    assert interpolator.iterations == 4
    assert interpolator.window == 5

    with pytest.raises(ValueError, match="smoothing passes"):
        interpolator.iterations = 0
    with pytest.raises(ValueError, match="smoothing window"):
        interpolator.window = 2
    assert interpolator.iterations == 4
    assert interpolator.window == 5


def test_smoothing_moves_an_interior_waypoint_and_pins_the_ends():
    path = [[0.0, 0.0], [1.0, 3.0], [2.0, 0.0]]
    smoothed = MovingAverageInterpolator(iterations=1, window=3).interpolate(
        path
    )

    assert list(smoothed[0]) == [0.0, 0.0]
    assert list(smoothed[-1]) == [2.0, 0.0]
    assert smoothed[1][1] < 3.0


def test_a_path_shorter_than_the_window_passes_through_untouched():
    path = [[0.0, 0.0], [1.0, 1.0]]
    smoothed = MovingAverageInterpolator(window=5).interpolate(path)
    assert [list(point) for point in smoothed] == path


# ---------------------------------------------------------------------
# The Dubins primitive
# ---------------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
def test_a_turning_radius_that_is_not_positive_is_refused(radius):
    with pytest.raises(ValueError, match="turning radius is"):
        DubinsPrimitive(radius)


def test_the_turning_radius_revalidates_when_it_is_replaced():
    primitive = DubinsPrimitive(2.0)
    primitive.turning_radius = 0.5
    assert primitive.turning_radius == pytest.approx(0.5, abs=METERS)

    with pytest.raises(ValueError, match="turning radius is 0"):
        primitive.turning_radius = 0.0
    assert primitive.turning_radius == pytest.approx(0.5, abs=METERS)


def test_a_state_shorter_than_a_position_cannot_be_judged_feasible():
    with pytest.raises(
        ValueError, match=r"state needs at least 2 element\(s\), got 1"
    ):
        DubinsPrimitive(1.0).is_feasible([1.0])


def test_a_state_carrying_a_nan_is_refused_rather_than_called_feasible():
    """Deviation A-21: a NaN compares false against every bound, so an
    unchecked state would be reported feasible."""
    with pytest.raises(ValueError, match="state is NaN, which is not finite"):
        DubinsPrimitive(1.0).is_feasible([float("nan"), 0.0])


def test_a_turn_tighter_than_the_radius_is_reported_infeasible():
    primitive = DubinsPrimitive(2.0)
    assert primitive.is_feasible([0.0, 0.0, 0.0, 1.0, 0.1])
    assert not primitive.is_feasible([0.0, 0.0, 0.0, 1.0, 10.0])


# ---------------------------------------------------------------------
# The vehicle
# ---------------------------------------------------------------------


def test_a_vehicle_pose_component_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="x is NaN, which is not finite"):
        DubinsVehicle(float("nan"), 0.0, 0.0)


def test_a_limit_box_no_command_could_sit_in_is_refused():
    with pytest.raises(
        ValueError, match="minimum speed is 2, which is outside"
    ):
        DubinsVehicle(max_speed=1.0, min_speed=2.0)


def test_each_speed_limit_accepts_a_value_and_refuses_an_inconsistent_one():
    vehicle = DubinsVehicle(max_speed=5.0, min_speed=0.0)
    vehicle.max_speed = 8.0
    vehicle.min_speed = -1.0
    assert vehicle.max_speed == pytest.approx(8.0, abs=METERS)
    assert vehicle.min_speed == pytest.approx(-1.0, abs=METERS)

    with pytest.raises(ValueError, match="minimum speed is"):
        vehicle.min_speed = 99.0
    assert vehicle.min_speed == pytest.approx(-1.0, abs=METERS)


def test_the_two_names_for_the_acceleration_bound_stay_in_step():
    """Deviation A-20 holds the five bounds as one limit set, so
    max_acceleration and max_speed_rate name the same number."""
    vehicle = DubinsVehicle()
    vehicle.max_acceleration = 3.0
    assert vehicle.max_speed_rate == pytest.approx(3.0, abs=METERS)

    vehicle.max_speed_rate = 6.0
    assert vehicle.max_acceleration == pytest.approx(6.0, abs=METERS)


def test_the_two_names_for_the_turn_rate_bound_stay_in_step():
    vehicle = DubinsVehicle()
    vehicle.max_turn_rate_dot = 4.0
    assert vehicle.max_turn_rate_change == pytest.approx(4.0, abs=RADIANS)

    vehicle.max_turn_rate_change = 7.0
    assert vehicle.max_turn_rate_dot == pytest.approx(7.0, abs=RADIANS)


def test_the_turn_rate_bound_accepts_a_value_and_reports_it_back():
    vehicle = DubinsVehicle(max_turn_rate=1.0)
    vehicle.max_turn_rate = 2.5
    assert vehicle.max_turn_rate == pytest.approx(2.5, abs=RADIANS)


def test_a_commanded_speed_outside_the_band_is_refused():
    vehicle = DubinsVehicle(max_speed=5.0)
    vehicle.speed = 4.0
    assert vehicle.speed == pytest.approx(4.0, abs=METERS)

    with pytest.raises(ValueError, match="speed is 99, which is outside"):
        vehicle.speed = 99.0
    assert vehicle.speed == pytest.approx(4.0, abs=METERS)


def test_a_commanded_turn_rate_that_is_not_finite_is_refused():
    vehicle = DubinsVehicle()
    vehicle.turn_rate = 0.5
    with pytest.raises(
        ValueError, match="turn rate is NaN, which is not finite"
    ):
        vehicle.turn_rate = float("nan")
    assert vehicle.turn_rate == pytest.approx(0.5, abs=RADIANS)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_a_step_over_a_non_positive_interval_is_refused(dt):
    with pytest.raises(ValueError, match="elapsed interval is"):
        DubinsVehicle().step(1.0, 0.0, dt)


def test_a_step_commanded_with_a_nan_is_refused():
    with pytest.raises(
        ValueError, match="commanded speed is NaN, which is not finite"
    ):
        DubinsVehicle().step(float("nan"), 0.0, 0.1)


def test_a_step_moves_the_vehicle_along_its_heading():
    vehicle = DubinsVehicle(max_acceleration=1000.0)
    vehicle.step(1.0, 0.0, 0.5)
    assert vehicle.x > 0.0
    assert vehicle.y == pytest.approx(0.0, abs=METERS)
    assert vehicle.heading == pytest.approx(0.0, abs=RADIANS)


def test_resetting_places_the_vehicle_and_brings_it_to_rest():
    vehicle = DubinsVehicle(max_acceleration=1000.0)
    vehicle.step(1.0, 0.5, 0.5)

    vehicle.reset(2.0, -3.0, 1.0)

    assert vehicle.pose == pytest.approx((2.0, -3.0, 1.0), abs=METERS)
    assert vehicle.speed == pytest.approx(0.0, abs=METERS)
    assert vehicle.turn_rate == pytest.approx(0.0, abs=RADIANS)


def test_resetting_to_a_pose_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="x is NaN, which is not finite"):
        DubinsVehicle().reset(float("nan"), 0.0, 0.0)


def test_inverse_kinematics_reports_the_speed_and_turn_it_needs():
    vehicle = DubinsVehicle()
    command = vehicle.inverse_kinematics(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1.0, 1.0
    )
    assert command[0] == pytest.approx(1.0, abs=METERS)
    assert command[1] == pytest.approx(math.pi / 4.0, abs=RADIANS)


def test_inverse_kinematics_refuses_a_segment_of_zero_duration():
    with pytest.raises(ValueError, match="segment duration is 0"):
        DubinsVehicle().inverse_kinematics(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.0
        )


def test_the_vehicle_refuses_to_judge_a_state_it_cannot_read():
    with pytest.raises(
        ValueError, match=r"state needs at least 2 element\(s\), got 1"
    ):
        DubinsVehicle().is_feasible([1.0])


# ---------------------------------------------------------------------
# Sampling planners
# ---------------------------------------------------------------------


@pytest.mark.parametrize("planner", [RRTPlanner, SSTPlanner])
def test_a_sampling_bound_that_is_not_a_pair_is_refused(planner):
    with pytest.raises(
        ValueError,
        match=r"each bound needs a low and a high, got 1 value\(s\)",
    ):
        planner(free_space(), [[0.0]])


@pytest.mark.parametrize("planner", [RRTPlanner, SSTPlanner])
def test_a_per_axis_step_with_a_negative_component_is_refused(planner):
    with pytest.raises(ValueError, match="step_size must be positive"):
        planner(free_space(), BOUNDS_2D, step_size=[1.0, -1.0])


def test_a_planner_refuses_a_goal_of_a_different_dimension():
    planner = RRTPlanner(free_space(), BOUNDS_2D, seed=1)
    with pytest.raises(ValueError, match="goal has dimension 2, expected 3"):
        planner.plan([0.0, 0.0, 0.0], [1.0, 1.0])


def test_a_planner_refuses_a_start_that_is_not_a_sequence():
    planner = RRTPlanner(free_space(), BOUNDS_2D, seed=1)
    with pytest.raises(TypeError):
        planner.plan("origin", [1.0, 1.0])


def test_measuring_between_states_of_different_length_is_refused():
    planner = RRTPlanner(free_space(), BOUNDS_2D, seed=1)
    with pytest.raises(ValueError, match="state has dimension 1, expected 2"):
        planner.distance([0.0, 0.0], [1.0])


def test_a_planner_reports_the_bounds_and_the_step_it_was_built_with():
    planner = RRTPlanner(
        free_space(), BOUNDS_2D, step_size=[0.5, 0.25], seed=1
    )
    assert planner.bounds == BOUNDS_2D
    assert list(planner.step_size) == [0.5, 0.25]
    assert planner.occupancy is not None


def test_a_scalar_step_is_broadcast_over_every_axis():
    planner = SSTPlanner(free_space(), BOUNDS_2D, step_size=0.75, seed=1)
    assert list(planner.step_size) == [0.75, 0.75]


# ---------------------------------------------------------------------
# Grid and graph search
# ---------------------------------------------------------------------


def test_an_unknown_grid_connectivity_is_refused_by_name():
    with pytest.raises(
        ValueError,
        match="grid_type must be 'euclidean' or 'manhattan'",
    ):
        AStar(np.zeros((3, 3), dtype=int), "diagonal")


def test_measuring_to_a_cell_off_the_edge_of_the_grid_is_a_key_error():
    planner = AStarPlanner(ManhattanGrid((5, 5)))
    with pytest.raises(KeyError, match="the node is outside the map"):
        planner.distance((99, 99), (0, 0))
    with pytest.raises(KeyError, match="the node is outside the map"):
        planner.heuristic((0, 0), (99, 99))


def test_measuring_over_an_edge_the_graph_does_not_carry_is_refused():
    planner = AStarPlanner(straight_graph())
    with pytest.raises(KeyError, match="unknown edge identifier"):
        planner.distance(0, 99)


def test_a_node_that_is_not_an_index_tuple_is_refused():
    planner = AStarPlanner(ManhattanGrid((5, 5)))
    with pytest.raises(TypeError):
        planner.distance("here", (0, 0))


def test_a_search_from_an_occupied_cell_reports_why_it_declined():
    grid = ManhattanGrid((5, 5))
    grid.data[2, :] = 1
    planner = AStarPlanner(grid)

    assert planner.plan((2, 0), (4, 4)) is None
    assert str(planner.last_failure) == "the start state is occupied"
    assert planner.last_failure.is_retryable is False


def test_a_diagnostic_search_from_an_occupied_cell_expands_nothing():
    grid = ManhattanGrid((5, 5))
    grid.data[2, :] = 1
    planner = AStarPlanner(grid)

    path, expanded, came_from = planner.plan_with_diagnostics((2, 0), (4, 4))

    assert path is None
    assert expanded == []
    assert came_from == {}


def test_a_goal_behind_a_wall_is_reported_unreachable():
    grid = ManhattanGrid((5, 5))
    grid.data[2, :] = 1
    planner = AStarPlanner(grid)

    assert planner.plan((0, 0), (4, 4)) is None
    assert str(planner.last_failure) == "no path exists"
    assert planner.last_failure.is_retryable is False


def test_a_search_outside_the_grid_is_a_distinct_answer():
    """Deviation A-15: a cell off the edge is not an unreachable one."""
    planner = AStarPlanner(ManhattanGrid((5, 5)))

    assert planner.plan((99, 99), (0, 0)) is None
    assert str(planner.last_failure) == "the start state is outside the map"


def test_an_exhausted_expansion_budget_is_reported_as_retryable():
    """FR-SAFE-02: a bounded search separates budget from impossibility."""
    planner = AStarPlanner(ManhattanGrid((6, 6)))
    planner.max_expansions = 1

    assert planner.plan((0, 0), (5, 5)) is None
    assert str(planner.last_failure) == "the search budget ran out"
    assert planner.last_failure.is_retryable is True


def test_a_successful_search_clears_the_reason_and_reports_its_cost():
    planner = AStarPlanner(ManhattanGrid((6, 6)))
    path = planner.plan((0, 0), (5, 5))

    assert path is not None
    assert planner.last_failure is None
    assert planner.last_cost == pytest.approx(10.0, abs=METERS)
    assert planner.last_expanded > 0


def test_collinear_waypoints_are_collapsed_unless_the_caller_says_not_to():
    grid = ManhattanGrid((6, 6))
    simplified = AStarPlanner(grid).plan((0, 0), (5, 5))
    verbatim = AStarPlanner(grid, simplify_path=False).plan((0, 0), (5, 5))

    assert len(simplified) < len(verbatim)
    assert simplified[0] == verbatim[0]
    assert simplified[-1] == verbatim[-1]


def test_the_discrete_base_measures_with_the_graph_it_was_given():
    planner = DiscretePlanner(ManhattanGrid((3, 3)))
    assert planner.distance((0, 0), (1, 0)) == pytest.approx(1.0, abs=METERS)
    assert planner.heuristic((0, 0), (2, 2)) > 0.0


# ---------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------


def test_a_graph_with_no_node_positions_cannot_be_routed_over():
    class Unpositioned:
        nodes = [1, 2]
        edges = [(1, 2, 1.0)]

    with pytest.raises(
        TypeError, match="graph must expose nodes, edges and position"
    ):
        RouteRouter(Unpositioned())


def test_a_route_reports_both_projections_and_their_distances():
    router = RouteRouter(straight_graph())
    result = router.plan([0.0, 0.1], [2.0, 0.1])

    assert result.path == [0, 1, 2]
    assert result.start_node == 0
    assert result.goal_node == 2
    assert result.start_distance == pytest.approx(0.1, abs=METERS)
    assert result.goal_distance == pytest.approx(0.1, abs=METERS)
    assert router.last_failure is None


def test_a_position_outside_the_activation_radius_joins_no_route():
    router = RouteRouter(straight_graph(), activation_radius=0.01)

    assert router.plan([50.0, 50.0], [2.0, 0.0]) is None
    assert str(router.last_failure) == "the start state is outside the map"


# ---------------------------------------------------------------------
# Trajectory pruning and optimization
# ---------------------------------------------------------------------


def test_a_scalar_step_size_is_refused_by_the_pruner():
    with pytest.raises(
        ValueError, match="step_size must be a non-empty 1-D array"
    ):
        TrajectoryPruner(free_space(), 1.0)


def test_an_empty_step_size_is_refused_by_the_pruner():
    with pytest.raises(ValueError, match=r"got shape \(0,\)"):
        TrajectoryPruner(free_space(), [])


def test_a_step_size_component_that_is_not_positive_is_refused():
    with pytest.raises(
        ValueError, match="step_size elements must be strictly positive"
    ):
        TrajectoryPruner(free_space(), [1.0, -1.0])


def test_pruning_an_empty_path_gives_an_empty_path():
    pruner = TrajectoryPruner(free_space(), [1.0, 1.0])
    assert list(pruner.prune([])) == []


def test_a_straight_run_of_waypoints_prunes_to_its_endpoints():
    pruner = TrajectoryPruner(free_space(), [1.0, 1.0])
    path = [[float(x), 0.0] for x in range(5)]

    kept = pruner.prune(path)

    assert len(kept) == 2
    assert list(kept[0]) == [0.0, 0.0]
    assert list(kept[-1]) == [4.0, 0.0]


def test_the_optimizer_names_its_five_default_cost_terms():
    optimizer = TrajectoryOptimizer(free_space(), cruise_speed=1.0)
    assert [term.name for term in optimizer.cost_terms] == [
        "time",
        "deviation",
        "velocity",
        "collision",
        "dynamics",
    ]


def test_a_default_cost_term_cannot_be_handed_back_as_a_custom_one():
    """The five defaults are an enum inside the crate, so only their
    names cross the boundary and passing one back is refused."""
    occupancy = free_space()
    named = TrajectoryOptimizer(occupancy, 1.0).cost_terms
    optimizer = TrajectoryOptimizer(
        occupancy, 1.0, max_iter=2, cost_terms=named
    )

    with pytest.raises(TypeError, match="is not callable"):
        optimizer.optimize([[0.0, 0.0], [1.0, 0.0]])


def test_the_optimizer_reports_the_settings_it_was_built_with():
    occupancy = free_space()
    optimizer = TrajectoryOptimizer(
        occupancy, cruise_speed=2.0, max_iter=7, ftol=1e-5
    )

    assert optimizer.occupancy is occupancy
    assert optimizer.cruise_speed == pytest.approx(2.0, abs=METERS)
    assert optimizer.max_iter == 7
    assert optimizer.ftol == pytest.approx(1e-5, abs=0.0, rel=1e-12)


def test_the_recorded_solver_method_reads_and_writes():
    """Deviation A-08 replaced the solver, so the name is recorded and
    not acted on; the attribute still answers."""
    optimizer = TrajectoryOptimizer(free_space(), 1.0)
    assert optimizer.method == "L-BFGS-B"
    optimizer.method = "SLSQP"
    assert optimizer.method == "SLSQP"


def test_an_exhausted_iteration_budget_is_reported_on_the_result():
    optimizer = TrajectoryOptimizer(free_space(), 1.0, max_iter=1)
    result = optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert result.optimizer_success is False
    assert result.optimizer_status_code == 1
    assert result.optimizer_status_text == "iteration budget exhausted"
    assert result.optimizer_iteration_count == 1


def test_a_result_built_with_no_arguments_matches_another_empty_one():
    assert TrajectoryResult() == TrajectoryResult()


def test_a_result_compares_unequal_against_something_of_another_type():
    assert (TrajectoryResult() == 5) is False


def test_a_result_field_reads_back_after_it_is_written():
    result = TrajectoryResult()
    result.cost = 3.0
    assert result.cost == pytest.approx(3.0, abs=0.0, rel=1e-12)
    assert result != TrajectoryResult()


def test_a_result_prints_the_nine_fields_the_dataclass_printed():
    printed = repr(TrajectoryResult())
    assert printed.startswith("TrajectoryResult(states=[]")
    assert "optimizer_iteration_count=0)" in printed
    assert "turn_rates" not in printed


# ---------------------------------------------------------------------
# The grid wrapper and the second grid connectivity
# ---------------------------------------------------------------------


def test_a_grid_array_is_searched_under_either_connectivity():
    """A four-connected search cannot cut a corner, so it needs at
    least as many moves as an eight-connected one over the same grid."""
    cells = np.zeros((6, 6), dtype=int)
    cells[3, 0:4] = 1

    straight = AStar(cells, "manhattan")
    diagonal = AStar(cells, "euclidean")
    along = straight.search((0, 0), (5, 5))
    across = diagonal.search((0, 0), (5, 5))

    assert along is not None
    assert across is not None
    assert straight.last_failure is None
    assert diagonal.last_failure is None
    assert along[0] == across[0] == (0, 0)
    assert along[-1] == across[-1] == (5, 5)


def test_a_grid_of_solid_cells_refuses_the_start_it_was_given():
    blocked = AStar(np.ones((4, 4), dtype=int))

    assert blocked.search((0, 0), (3, 3)) is None
    assert str(blocked.last_failure) == "the start state is occupied"


def test_an_eight_connected_grid_measures_along_the_diagonal():
    planner = AStarPlanner(EuclideanGrid((7, 7)))

    assert planner.is_native is True
    assert planner.distance((0, 0), (1, 1)) == pytest.approx(
        math.sqrt(2.0), abs=METERS
    )
    assert planner.plan((0, 0), (6, 6)) is not None
    assert planner.last_cost == pytest.approx(6.0 * math.sqrt(2.0), abs=METERS)


def test_a_grid_too_small_to_have_an_interior_cell_still_searches():
    """Connectivity is read off an interior cell, and a grid with no
    interior falls back to the four-connected reading."""
    planner = AStarPlanner(ManhattanGrid((2, 2)))

    assert planner.is_native is True
    assert planner.plan((0, 0), (1, 1)) == [(0, 0), (1, 0), (1, 1)]


# ---------------------------------------------------------------------
# A Python subclass forwarding its arguments upward
# ---------------------------------------------------------------------


def test_a_subclass_may_call_super_init_with_the_arguments_it_took():
    """A compiled class constructs in __new__, so __init__ has to
    absorb what a subclass forwards rather than refuse it."""
    occupancy = free_space()

    class Pruner(TrajectoryPruner):
        def __init__(self, occupancy, step_size):
            super().__init__(occupancy, step_size)
            self.label = "mine"

    class Optimizer(TrajectoryOptimizer):
        def __init__(self, occupancy, cruise_speed):
            super().__init__(occupancy, cruise_speed)
            self.label = "mine"

    class Router(RouteRouter):
        def __init__(self, graph):
            super().__init__(graph)
            self.label = "mine"

    class Search(AStar):
        def __init__(self, cells):
            super().__init__(cells)
            self.label = "mine"

    assert Pruner(occupancy, [1.0, 1.0]).label == "mine"
    assert Optimizer(occupancy, 1.0).label == "mine"
    assert Router(straight_graph()).label == "mine"
    assert Search(np.zeros((3, 3), dtype=int)).label == "mine"


def test_a_cost_subclass_may_call_super_init_and_keep_its_own_state():
    class Scaled(PlannerCost):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

    model = Scaled(2.0)

    assert model.scale == pytest.approx(2.0, abs=METERS)
    assert model.distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        5.0, abs=METERS
    )


# ---------------------------------------------------------------------
# The attributes each planner publishes about itself
# ---------------------------------------------------------------------


def test_a_route_router_publishes_the_graph_and_the_radius():
    graph = straight_graph()
    router = RouteRouter(graph, activation_radius=2.0)

    assert router.graph is graph
    assert router.activation_radius == pytest.approx(2.0, abs=METERS)

    router.activation_radius = 5.0
    assert router.activation_radius == pytest.approx(5.0, abs=METERS)


def test_a_discrete_planner_publishes_the_graph_it_searches():
    graph = straight_graph()
    assert DiscretePlanner(graph).graph is graph


def test_a_continuous_planner_publishes_its_map_and_its_generator():
    occupancy = free_space()
    planner = ContinuousPlanner(occupancy, seed=3)

    assert planner.occupancy is occupancy
    assert planner.make_rng().random() == pytest.approx(
        np.random.default_rng(3).random(), abs=0.0, rel=0.0
    )


def test_a_continuous_planner_publishes_through_its_sink():
    received = []
    planner = ContinuousPlanner(free_space(), publisher=received.append)
    snapshot = PlannerTelemetry("RRT*", "done", 1, 2, 0.0)

    planner.publish_telemetry(snapshot)

    assert received == [snapshot]


def test_a_default_cost_term_prints_the_name_it_carries():
    term = TrajectoryOptimizer(free_space(), 1.0).cost_terms[0]
    assert term.name == "time"
    assert repr(term) == 'DefaultCostTerm(name="time")'
