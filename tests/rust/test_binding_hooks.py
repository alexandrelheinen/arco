"""The escape hatches every planner policy keeps for a Python callable.

ADR-004 splits each hook into an enum with a native variant per built-in
policy and one ``Custom`` variant holding whatever the caller injected.
The rest of the suite plans with the defaults, so the native variants
are well covered and the escape hatches are not. These tests select the
``Custom`` variant of every hook the binding exposes.

Two things are asserted about each hook, because either one alone
proves nothing. The callable is counted, so a test cannot pass against
a binding that ignored it; and the outcome is compared against what the
hook makes true, so a test cannot pass against a binding that called it
and threw the answer away.
"""

from __future__ import annotations

import numpy as np
import pytest

from arco._arco import (
    AStarPlanner,
    PlannerCost,
    RouteRouter,
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)
from arco.mapping import CartesianGraph, KDTreeOccupancy, ManhattanGrid
from arco.planning.continuous.telemetry import PlannerTelemetry

# Comparison tolerances for the assertions below, in meters and seconds.
METERS = 1e-12
SECONDS = 1e-12

BOUNDS_2D = [(0.0, 10.0), (0.0, 10.0)]
SAMPLING_PLANNERS = [RRTPlanner, SSTPlanner]


def free_space():
    """An occupancy whose only obstacle sits far outside the bounds."""
    return KDTreeOccupancy([[50.0, 50.0]], clearance=0.3)


def solid_wall():
    """A wall at x = 5 that spans the whole sampling range and beyond."""
    points = [[5.0, y] for y in np.arange(-2.0, 12.1, 0.2)]
    return KDTreeOccupancy(points, clearance=0.4)


def straight_graph():
    """Three nodes in a row, one meter apart, joined by two edges."""
    graph = CartesianGraph()
    for node, x in enumerate([0.0, 1.0, 2.0]):
        graph.add_node(node, x, 0.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return graph


class CountingCost(PlannerCost):
    """A metric that answers a constant and counts what it was asked."""

    def __init__(self, answer=7.0):
        super().__init__()
        self.answer = answer
        self.distance_calls = 0
        self.heuristic_calls = 0

    def distance(self, state_a, state_b):
        self.distance_calls += 1
        return self.answer

    def heuristic(self, state_a, state_b):
        self.heuristic_calls += 1
        return self.answer


class EuclideanCost(PlannerCost):
    """A usable metric that records how often a planner consulted it."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def distance(self, state_a, state_b):
        self.calls += 1
        return float(np.linalg.norm(np.asarray(state_a) - np.asarray(state_b)))

    def heuristic(self, state_a, state_b):
        return self.distance(state_a, state_b)


# ---------------------------------------------------------------------
# is_native, the property FR-PERF-03 asks callers to be able to check
# ---------------------------------------------------------------------


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_a_planner_built_from_defaults_runs_entirely_in_rust(
    planner_type,
):
    planner = planner_type(free_space(), BOUNDS_2D, seed=1)
    assert planner.is_native is True


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
@pytest.mark.parametrize(
    "hook", ["sampler", "steerer", "segment_free", "cost"]
)
def test_any_injected_hook_takes_the_planner_off_the_native_path(
    planner_type, hook
):
    injected = {hook: EuclideanCost() if hook == "cost" else _accept}
    planner = planner_type(free_space(), BOUNDS_2D, seed=1, **injected)
    assert planner.is_native is False


def _accept(*_arguments):
    """A hook shaped to satisfy every keyword the parametrization uses."""
    return True


def test_an_occupancy_this_binding_cannot_rebuild_is_not_native():
    planner = RRTPlanner(_OpenField(), BOUNDS_2D, seed=1)
    assert planner.is_native is False


def test_a_subclass_that_replaces_the_metric_is_not_native():
    class Rescaled(RRTPlanner):
        def distance(self, state_a, state_b):
            return 1.0

    planner = Rescaled(free_space(), BOUNDS_2D, seed=1)
    assert planner.is_native is False


# ---------------------------------------------------------------------
# sampler=
# ---------------------------------------------------------------------


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_an_injected_sampler_is_the_only_source_of_new_states(
    planner_type,
):
    """A sampler pinned to the start, with the goal bias switched off,
    leaves the planner nothing to grow toward, so the plan has to fail."""
    calls = []

    def pinned(rng):
        calls.append(rng)
        return np.array([0.5, 0.5])

    planner = planner_type(
        free_space(),
        BOUNDS_2D,
        max_sample_count=200,
        goal_bias=0.0,
        sampler=pinned,
        seed=3,
    )

    assert planner.plan([0.5, 0.5], [9.0, 9.0]) is None
    assert len(calls) > 0
    assert str(planner.last_failure) == "the search budget ran out"


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_a_sampler_that_draws_normally_still_finds_a_path(planner_type):
    calls = []

    def drawn(rng):
        calls.append(1)
        return rng.uniform([0.0, 0.0], [10.0, 10.0])

    planner = planner_type(
        free_space(),
        BOUNDS_2D,
        max_sample_count=1000,
        sampler=drawn,
        seed=3,
    )

    assert planner.plan([0.5, 0.5], [9.0, 9.0]) is not None
    assert len(calls) > 0


def test_the_public_sample_helper_asks_the_injected_sampler():
    calls = []

    def fixed(rng):
        calls.append(rng)
        return np.array([2.0, 3.0])

    planner = RRTPlanner(free_space(), BOUNDS_2D, sampler=fixed, seed=1)
    generator = np.random.default_rng(0)

    drawn = planner.sample(generator)

    assert list(drawn) == [2.0, 3.0]
    assert calls == [generator]


def test_the_public_sample_helper_draws_in_bounds_without_a_hook():
    planner = RRTPlanner(free_space(), BOUNDS_2D, seed=1)
    drawn = planner.sample(np.random.default_rng(0))
    assert 0.0 <= drawn[0] <= 10.0
    assert 0.0 <= drawn[1] <= 10.0


# ---------------------------------------------------------------------
# steerer=
# ---------------------------------------------------------------------


def test_an_injected_steerer_decides_how_far_one_extension_reaches():
    """A steerer that jumps the whole way makes the tree reach the goal
    in a handful of extensions, where the default step of 0.2 meters
    needs tens of them."""
    calls = []

    def teleport(from_state, to_state):
        calls.append((tuple(from_state), tuple(to_state)))
        return np.asarray(to_state, dtype=float)

    occupancy = free_space()
    settings = dict(max_sample_count=400, step_size=0.2, goal_bias=0.3, seed=5)
    jumped = RRTPlanner(
        occupancy, BOUNDS_2D, steerer=teleport, **settings
    ).plan([0.5, 0.5], [9.0, 9.0])
    stepped = RRTPlanner(occupancy, BOUNDS_2D, **settings).plan(
        [0.5, 0.5], [9.0, 9.0]
    )

    assert len(calls) > 0
    assert jumped is not None
    assert stepped is not None
    assert len(jumped) < len(stepped)


def test_the_public_steer_helper_returns_what_the_hook_returned():
    calls = []

    def sideways(from_state, to_state):
        calls.append((tuple(from_state), tuple(to_state)))
        return np.array([-1.0, -1.0])

    planner = SSTPlanner(free_space(), BOUNDS_2D, steerer=sideways, seed=1)

    stepped = planner.steer([0.0, 0.0], [5.0, 0.0])

    assert list(stepped) == [-1.0, -1.0]
    assert calls == [((0.0, 0.0), (5.0, 0.0))]


def test_the_public_steer_helper_takes_one_bounded_step_by_default():
    planner = RRTPlanner(free_space(), BOUNDS_2D, step_size=1.0, seed=1)
    stepped = planner.steer([0.0, 0.0], [5.0, 0.0])
    assert list(stepped) == [1.0, 0.0]


# ---------------------------------------------------------------------
# segment_free=
# ---------------------------------------------------------------------


def test_an_injected_segment_check_replaces_the_collision_test():
    """The wall spans the whole sampling range, so the built-in check
    cannot get across it and a hook that accepts every segment can."""
    calls = []

    def always_free(from_state, to_state):
        calls.append((tuple(from_state), tuple(to_state)))
        return True

    occupancy = solid_wall()
    settings = dict(max_sample_count=1500, seed=11)
    blocked = RRTPlanner(occupancy, BOUNDS_2D, **settings).plan(
        [1.0, 5.0], [9.0, 5.0]
    )
    ignored = RRTPlanner(
        occupancy, BOUNDS_2D, segment_free=always_free, **settings
    ).plan([1.0, 5.0], [9.0, 5.0])

    assert blocked is None
    assert ignored is not None
    assert len(calls) > 0


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_a_segment_check_that_refuses_everything_finds_no_path(
    planner_type,
):
    calls = []

    def refuse(from_state, to_state):
        calls.append(1)
        return False

    planner = planner_type(
        free_space(),
        BOUNDS_2D,
        max_sample_count=100,
        segment_free=refuse,
        seed=1,
    )

    assert planner.plan([0.5, 0.5], [9.0, 9.0]) is None
    assert len(calls) > 0


def test_the_public_segment_helper_answers_what_the_hook_answered():
    calls = []

    def refuse(from_state, to_state):
        calls.append((tuple(from_state), tuple(to_state)))
        return False

    planner = RRTPlanner(free_space(), BOUNDS_2D, segment_free=refuse, seed=1)

    assert planner.is_segment_free([0.0, 0.0], [1.0, 1.0]) is False
    assert calls == [((0.0, 0.0), (1.0, 1.0))]


def test_the_public_segment_helper_uses_the_map_without_a_hook():
    planner = RRTPlanner(free_space(), BOUNDS_2D, seed=1)
    assert planner.is_segment_free([0.0, 0.0], [1.0, 1.0]) is True


# ---------------------------------------------------------------------
# cost=
# ---------------------------------------------------------------------


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_the_public_metric_helpers_report_the_injected_model(
    planner_type,
):
    model = CountingCost(answer=7.0)
    planner = planner_type(free_space(), BOUNDS_2D, cost=model, seed=1)

    assert planner.distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        7.0, abs=METERS
    )
    assert planner.heuristic([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        7.0, abs=METERS
    )
    assert model.distance_calls == 1
    assert model.heuristic_calls == 1


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_a_planner_measures_with_the_injected_model_while_it_plans(
    planner_type,
):
    model = EuclideanCost()
    planner = planner_type(
        free_space(),
        BOUNDS_2D,
        max_sample_count=200,
        cost=model,
        seed=4,
    )

    path = planner.plan([0.5, 0.5], [9.0, 9.0])

    assert path is not None
    assert model.calls > 0


def test_the_native_metric_normalizes_by_the_step_size():
    planner = RRTPlanner(free_space(), BOUNDS_2D, step_size=[2.0, 2.0], seed=1)
    assert planner.distance([0.0, 0.0], [6.0, 8.0]) == pytest.approx(
        5.0, abs=METERS
    )


# ---------------------------------------------------------------------
# A subclass replacing distance or heuristic, deviation A-24
# ---------------------------------------------------------------------


def test_a_subclass_metric_is_the_one_the_planner_measures_with():
    class Counted(RRTPlanner):
        calls = 0

        def distance(self, state_a, state_b):
            type(self).calls += 1
            return float(
                np.linalg.norm(np.asarray(state_a) - np.asarray(state_b))
            )

    planner = Counted(free_space(), BOUNDS_2D, max_sample_count=150, seed=2)

    path = planner.plan([0.5, 0.5], [9.0, 9.0])

    assert path is not None
    assert Counted.calls > 0


def test_a_subclass_metric_also_drives_the_reported_tree():
    class Counted(SSTPlanner):
        calls = 0

        def distance(self, state_a, state_b):
            type(self).calls += 1
            return float(
                np.linalg.norm(np.asarray(state_a) - np.asarray(state_b))
            )

    planner = Counted(free_space(), BOUNDS_2D, max_sample_count=120, seed=2)

    nodes, parent, _path = planner.get_tree([0.5, 0.5], [9.0, 9.0])

    assert len(nodes) > 1
    assert len(parent) == len(nodes)
    assert Counted.calls > 0


# ---------------------------------------------------------------------
# publisher=
# ---------------------------------------------------------------------


@pytest.mark.parametrize("planner_type", SAMPLING_PLANNERS)
def test_an_injected_sink_receives_the_progress_snapshots(planner_type):
    received = []
    planner = planner_type(
        free_space(),
        BOUNDS_2D,
        max_sample_count=200,
        publisher=received.append,
        seed=2,
    )

    planner.plan([0.5, 0.5], [9.0, 9.0])

    assert len(received) > 0
    assert all(isinstance(snapshot, PlannerTelemetry) for snapshot in received)
    assert received[0].max_iterations == 200


def test_publishing_by_hand_reaches_the_injected_sink():
    received = []
    planner = RRTPlanner(
        free_space(), BOUNDS_2D, publisher=received.append, seed=1
    )
    snapshot = PlannerTelemetry("RRT*", "done", 1, 2, 0.0)

    planner.publish_telemetry(snapshot)

    assert received == [snapshot]


def test_a_sink_that_fails_mid_plan_does_not_abandon_the_plan():
    """Python's write_telemetry swallowed every error for the same
    reason: a loading screen that has gone away is not a failure."""

    def broken(_snapshot):
        raise RuntimeError("sink down")

    planner = RRTPlanner(
        free_space(),
        BOUNDS_2D,
        max_sample_count=300,
        publisher=broken,
        seed=2,
    )

    assert planner.plan([0.5, 0.5], [9.0, 9.0]) is not None


def test_publishing_by_hand_surfaces_what_the_sink_raised():
    def broken(_snapshot):
        raise RuntimeError("sink down")

    planner = RRTPlanner(free_space(), BOUNDS_2D, publisher=broken, seed=1)

    with pytest.raises(RuntimeError, match="sink down"):
        planner.publish_telemetry(PlannerTelemetry("RRT*", "done", 1, 2, 0.0))


# ---------------------------------------------------------------------
# A caller's own occupancy, the PyOccupancy adapter
# ---------------------------------------------------------------------


class _OpenField:
    """An occupancy of a shape this binding cannot rebuild natively."""

    dimension = 2
    clearance = 0.0

    def __init__(self):
        self.occupied_calls = 0
        self.nearest_calls = 0
        self.segment_calls = 0

    def is_occupied(self, point):
        self.occupied_calls += 1
        return False

    def nearest_obstacle(self, point):
        self.nearest_calls += 1
        return 1e6, np.array([1e6, 1e6])

    def segment_free(self, from_state, to_state):
        self.segment_calls += 1
        return True


class _SolidBlock(_OpenField):
    """An occupancy that calls every point occupied."""

    def is_occupied(self, point):
        self.occupied_calls += 1
        return True


def test_a_caller_supplied_map_is_queried_through_the_interpreter():
    occupancy = _OpenField()
    planner = RRTPlanner(occupancy, BOUNDS_2D, max_sample_count=300, seed=4)

    path = planner.plan([0.5, 0.5], [9.0, 9.0])

    assert path is not None
    assert occupancy.occupied_calls > 0


def test_a_caller_supplied_map_that_blocks_everything_finds_no_path():
    occupancy = _SolidBlock()
    planner = RRTPlanner(occupancy, BOUNDS_2D, max_sample_count=100, seed=4)

    assert planner.plan([0.5, 0.5], [9.0, 9.0]) is None
    assert occupancy.occupied_calls > 0


def test_a_pruner_keeps_the_caller_supplied_map_it_was_given():
    occupancy = _OpenField()
    pruner = TrajectoryPruner(occupancy, [1.0, 1.0])
    assert pruner.occupancy is occupancy


# ---------------------------------------------------------------------
# heuristic= and a caller's own graph
# ---------------------------------------------------------------------


def test_an_injected_heuristic_steers_how_much_the_search_expands():
    """All three heuristics below are consistent with the same optimal
    cost, and the expansion counts order the way A* says they should:
    an inflated estimate expands least and a zero estimate expands the
    whole grid."""
    grid = ManhattanGrid((15, 15))
    grid.data[7, 0:13] = 1
    inflated_calls = []
    zero_calls = []

    def inflated(node, goal):
        inflated_calls.append(1)
        return 50.0 * (abs(node[0] - goal[0]) + abs(node[1] - goal[1]))

    def zero(node, goal):
        zero_calls.append(1)
        return 0.0

    native = AStarPlanner(grid)
    greedy = AStarPlanner(grid, heuristic=inflated)
    dijkstra = AStarPlanner(grid, heuristic=zero)
    for planner in (native, greedy, dijkstra):
        assert planner.plan((0, 0), (14, 14)) is not None

    assert len(inflated_calls) > 0
    assert len(zero_calls) > 0
    assert native.last_cost == greedy.last_cost == dijkstra.last_cost
    assert dijkstra.last_expanded == 15 * 15
    assert greedy.last_expanded < native.last_expanded
    assert native.last_expanded < dijkstra.last_expanded


def test_an_injected_heuristic_takes_the_search_off_the_native_path():
    planner = AStarPlanner(
        ManhattanGrid((5, 5)), heuristic=lambda node, goal: 0.0
    )
    assert planner.is_native is False


def test_a_grid_search_with_no_heuristic_hook_stays_native():
    assert AStarPlanner(ManhattanGrid((5, 5))).is_native is True


class _Chain:
    """A graph of a shape this binding has no native type for."""

    def __init__(self):
        self.links = {
            "a": ["b"],
            "b": ["a", "c"],
            "c": ["b", "d"],
            "d": ["c"],
        }
        self.neighbor_calls = 0
        self.distance_calls = 0
        self.heuristic_calls = 0
        self.occupied_calls = 0

    def neighbors(self, node):
        self.neighbor_calls += 1
        return self.links.get(node, [])

    def distance(self, node_a, node_b):
        self.distance_calls += 1
        return 1.0

    def heuristic(self, node_a, node_b):
        self.heuristic_calls += 1
        return 0.0

    def is_occupied(self, node):
        self.occupied_calls += 1
        return False


def test_a_caller_supplied_graph_is_searched_through_its_own_methods():
    graph = _Chain()
    planner = AStarPlanner(graph)

    path = planner.plan("a", "d")

    assert path == ["a", "b", "c", "d"]
    assert planner.is_native is False
    assert planner.last_cost == pytest.approx(3.0, abs=METERS)
    assert graph.neighbor_calls > 0
    assert graph.distance_calls > 0
    assert graph.heuristic_calls > 0
    assert graph.occupied_calls > 0


def test_a_caller_supplied_graph_reports_its_expansion_order():
    graph = _Chain()
    planner = AStarPlanner(graph)

    path, expanded, came_from = planner.plan_with_diagnostics("a", "d")

    assert path == ["a", "b", "c", "d"]
    assert expanded[0] == "a"
    assert came_from["d"] == "c"


def test_a_graph_publishing_no_heuristic_falls_back_to_its_distance():
    class Bare:
        def __init__(self):
            self.distance_calls = 0

        def neighbors(self, node):
            return {"a": ["b"], "b": ["a"]}.get(node, [])

        def distance(self, node_a, node_b):
            self.distance_calls += 1
            return 2.0

    graph = Bare()
    planner = AStarPlanner(graph)

    assert planner.plan("a", "b") == ["a", "b"]
    assert planner.last_cost == pytest.approx(2.0, abs=METERS)
    assert graph.distance_calls > 0


def test_a_subclass_heuristic_replaces_the_grid_metric():
    class Counted(AStarPlanner):
        calls = 0

        def heuristic(self, state_a, state_b):
            type(self).calls += 1
            return 0.0

    planner = Counted(ManhattanGrid((6, 6)))

    assert planner.is_native is False
    assert planner.plan((0, 0), (5, 5)) is not None
    assert Counted.calls > 0


# ---------------------------------------------------------------------
# RouteRouter planner=
# ---------------------------------------------------------------------


def test_an_injected_router_supplies_the_path_between_the_projections():
    calls = []

    class Direct:
        def plan(self, start_node, goal_node):
            calls.append((start_node, goal_node))
            return [start_node, goal_node]

    router = RouteRouter(straight_graph(), planner=Direct())

    result = router.plan([0.0, 0.1], [2.0, 0.1])

    assert calls == [(0, 2)]
    assert result.path == [0, 2]
    assert result.start_node == 0
    assert result.goal_node == 2


def test_an_injected_router_that_finds_nothing_reports_no_route():
    class Empty:
        def __init__(self):
            self.calls = 0

        def plan(self, start_node, goal_node):
            self.calls += 1
            return None

    injected = Empty()
    router = RouteRouter(straight_graph(), planner=injected)

    assert router.plan([0.0, 0.1], [2.0, 0.1]) is None
    assert injected.calls == 1
    assert str(router.last_failure) == "no path exists"


# ---------------------------------------------------------------------
# TrajectoryPruner steer=
# ---------------------------------------------------------------------


def test_an_injected_shortcut_check_decides_what_the_pruner_keeps():
    """The same zigzag prunes to its endpoints when the hook accepts
    every shortcut, and survives untouched when it refuses them all."""
    zigzag = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [4.0, 0.0],
    ]
    pruner = TrajectoryPruner(free_space(), [1.0, 1.0])
    accepted = []
    refused = []

    def accept(from_state, to_state):
        accepted.append((tuple(from_state), tuple(to_state)))
        return True

    def refuse(from_state, to_state):
        refused.append((tuple(from_state), tuple(to_state)))
        return False

    kept = pruner.prune(zigzag, steer=accept)
    whole = pruner.prune(zigzag, steer=refuse)

    assert len(accepted) > 0
    assert len(refused) > 0
    assert [list(point) for point in kept] == [
        [0.0, 0.0],
        [4.0, 0.0],
    ]
    assert [list(point) for point in whole] == zigzag


# ---------------------------------------------------------------------
# TrajectoryOptimizer cost_terms=, feasibility= and inverse_kinematics=
# ---------------------------------------------------------------------


def test_an_injected_cost_term_is_the_cost_the_solver_reports():
    """With one term summing the durations, the cost at the returned
    solution has to be that sum, which no default term would produce."""
    seen = []

    def total_duration(context):
        seen.append(context)
        return float(np.sum(context["durations"]))

    optimizer = TrajectoryOptimizer(
        free_space(), 1.0, cost_terms=[total_duration], max_iter=40
    )

    result = optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert len(seen) > 0
    assert result.cost == pytest.approx(sum(result.durations), abs=SECONDS)


def test_an_injected_cost_term_reads_the_context_python_terms_read():
    seen = []

    def inspecting(context):
        seen.append(context)
        return 1.0

    optimizer = TrajectoryOptimizer(
        free_space(),
        cruise_speed=2.0,
        cost_terms=[inspecting],
        max_iter=2,
    )
    optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    context = seen[0]
    assert context["dim"] == 2
    assert context["segment_count"] == 2
    assert context["cruise_speed"] == pytest.approx(2.0, abs=METERS)
    assert len(context["durations"]) == 2
    assert len(context["waypoints"]) == 3
    assert len(context["ref"]) == 3
    assert context["pts"].shape == (3, 2)


def test_the_injected_terms_are_the_ones_the_optimizer_reports():
    def term(context):
        return 0.0

    optimizer = TrajectoryOptimizer(free_space(), 1.0, cost_terms=[term])
    assert optimizer.cost_terms == [term]


def test_an_injected_feasibility_check_sees_every_state():
    accepted = []

    def accept(state):
        accepted.append(state)
        return True

    optimizer = TrajectoryOptimizer(free_space(), 1.0, max_iter=5)
    reference = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]

    result = optimizer.optimize(reference, feasibility=accept)

    assert result.is_feasible is True
    assert len(accepted) == len(result.states)
    assert len(accepted[0]) == 5


def test_an_injected_feasibility_check_that_refuses_marks_the_result():
    refusals = []

    def refuse(state):
        refusals.append(state)
        return False

    optimizer = TrajectoryOptimizer(free_space(), 1.0, max_iter=5)

    result = optimizer.optimize(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], feasibility=refuse
    )

    assert result.is_feasible is False
    assert len(refusals) > 0


def test_injected_inverse_kinematics_supplies_every_command():
    calls = []

    def kinematics(from_state, to_state, speed, duration):
        calls.append((speed, duration))
        return np.array([speed, 0.5])

    optimizer = TrajectoryOptimizer(free_space(), 1.0, max_iter=5)
    reference = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]

    result = optimizer.optimize(reference, inverse_kinematics=kinematics)

    assert len(calls) == len(result.commands)
    assert len(result.commands) == len(result.durations)
    for command in result.commands:
        assert command[1] == pytest.approx(0.5, abs=METERS)


def test_without_inverse_kinematics_the_turn_rate_is_published_apart():
    """The Python optimizer wrote a zero turn rate into every command
    and the real one reaches a caller as turn_rates."""
    optimizer = TrajectoryOptimizer(free_space(), 1.0, max_iter=5)

    result = optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert all(
        command[1] == pytest.approx(0.0, abs=METERS)
        for command in result.commands
    )
    assert len(result.turn_rates) == len(result.commands)


# ---------------------------------------------------------------------
# A hook that raises
# ---------------------------------------------------------------------


def raising_hook(failure):
    """Builds a hook of any arity that raises *failure* when called.

    Args:
        failure: The exception instance the hook raises.

    Returns:
        A callable accepting whatever arguments the hook is given.
    """

    def hook(*_arguments):
        raise failure

    return hook


@pytest.mark.parametrize(
    "hook, failure",
    [
        ("sampler", ValueError("sampler exploded")),
        ("steerer", KeyError("steerer exploded")),
        ("segment_free", TypeError("segment check exploded")),
    ],
)
def test_a_raising_planner_hook_keeps_its_own_exception_type(hook, failure):
    """FR-API-04: a caller that wrapped a plan in except ValueError
    keeps catching what it used to catch, so the exception a hook
    raised has to survive the trip through a trait that cannot carry
    one."""
    planner = RRTPlanner(
        free_space(),
        BOUNDS_2D,
        max_sample_count=50,
        seed=1,
        **{hook: raising_hook(failure)},
    )

    with pytest.raises(type(failure)) as raised:
        planner.plan([0.5, 0.5], [9.0, 9.0])
    assert "exploded" in str(raised.value)


def test_a_raising_cost_model_surfaces_its_own_exception():
    class Angry(PlannerCost):
        def distance(self, state_a, state_b):
            raise ArithmeticError("metric exploded")

        def heuristic(self, state_a, state_b):
            raise ArithmeticError("metric exploded")

    planner = SSTPlanner(
        free_space(),
        BOUNDS_2D,
        max_sample_count=50,
        cost=Angry(),
        seed=1,
    )

    with pytest.raises(ArithmeticError, match="metric exploded"):
        planner.plan([0.5, 0.5], [9.0, 9.0])


def test_a_raising_shortcut_check_surfaces_its_own_exception():
    def broken(from_state, to_state):
        raise KeyError("shortcut exploded")

    pruner = TrajectoryPruner(free_space(), [1.0, 1.0])

    with pytest.raises(KeyError, match="shortcut exploded"):
        pruner.prune([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], steer=broken)


def test_a_raising_cost_term_surfaces_its_own_exception():
    def broken(context):
        raise TypeError("term exploded")

    optimizer = TrajectoryOptimizer(
        free_space(), 1.0, cost_terms=[broken], max_iter=3
    )

    with pytest.raises(TypeError, match="term exploded"):
        optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_a_raising_graph_metric_surfaces_its_own_exception():
    class Angry:
        def neighbors(self, node):
            return {"a": ["b"], "b": ["a"]}.get(node, [])

        def distance(self, node_a, node_b):
            raise ValueError("graph metric exploded")

    with pytest.raises(ValueError, match="graph metric exploded"):
        AStarPlanner(Angry()).plan("a", "b")


def test_a_raising_occupancy_surfaces_its_own_exception():
    class Angry(_OpenField):
        def is_occupied(self, point):
            raise IndexError("map exploded")

    planner = RRTPlanner(Angry(), BOUNDS_2D, max_sample_count=20, seed=1)

    with pytest.raises(IndexError, match="map exploded"):
        planner.plan([0.5, 0.5], [9.0, 9.0])


# ---------------------------------------------------------------------
# The rest of the caller-supplied occupancy surface
# ---------------------------------------------------------------------


def test_the_optimizer_asks_a_caller_supplied_map_for_its_obstacles():
    """The collision term queries the nearest obstacle rather than cell
    occupancy, so this is the method a custom map has to answer."""
    occupancy = _OpenField()
    optimizer = TrajectoryOptimizer(occupancy, 1.0, max_iter=3)

    optimizer.optimize([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert occupancy.nearest_calls > 0


def test_the_pruner_asks_a_caller_supplied_map_about_each_shortcut():
    occupancy = _OpenField()
    pruner = TrajectoryPruner(occupancy, [1.0, 1.0])

    kept = pruner.prune([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]])

    assert occupancy.segment_calls > 0
    assert [list(point) for point in kept] == [
        [0.0, 0.0],
        [2.0, 0.0],
    ]


def test_a_map_that_refuses_every_shortcut_keeps_the_whole_path():
    class Congested(_OpenField):
        def segment_free(self, from_state, to_state):
            self.segment_calls += 1
            return False

    occupancy = Congested()
    path = [[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]]

    kept = TrajectoryPruner(occupancy, [1.0, 1.0]).prune(path)

    assert occupancy.segment_calls > 0
    assert [list(point) for point in kept] == path


def test_a_graph_whose_node_names_are_not_identifiers_keeps_its_methods():
    """A positioned graph is rebuilt natively only when its node names
    are identifiers the native graph can hold; anything else falls back
    to the Python adapter rather than being refused."""

    class Named:
        nodes = ["a", "b"]
        edges = [("a", "b", 1.0)]

        def __init__(self):
            self.neighbor_calls = 0

        def position(self, node):
            return [0.0, 0.0]

        def neighbors(self, node):
            self.neighbor_calls += 1
            return {"a": ["b"], "b": ["a"]}.get(node, [])

        def distance(self, node_a, node_b):
            return 1.0

    graph = Named()
    planner = AStarPlanner(graph)

    assert planner.is_native is False
    assert planner.plan("a", "b") == ["a", "b"]
    assert graph.neighbor_calls > 0
