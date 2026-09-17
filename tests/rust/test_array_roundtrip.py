"""Arrays cross the binding unchanged in shape and dtype (FR-API-03).

A compiled function that hands back a list, a memoryview, or an array of
the wrong width is a working function and a broken drop-in: the caller's
next line indexes it, reshapes it, or feeds it to numpy, and any of those
fails on the wrong type. Every assertion here is about the container the
caller receives rather than about the numbers inside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from arco.control import ActuatorArray, CircleBody, JointSpaceTracker
from arco.control.mpc import ReferencePath
from arco.mapping import KDTreeOccupancy


def _tracker() -> JointSpaceTracker:
    tracker = JointSpaceTracker(max_vel=1.0, max_acc=2.0)
    tracker.reset(np.zeros(3))
    return tracker


def test_a_returned_array_is_a_numpy_array_of_doubles() -> None:
    """The container is a real ndarray, not a list wearing its shape."""
    moved = _tracker().step(np.array([0.2, 0.0, 0.0]), 0.05)
    assert isinstance(moved, np.ndarray)
    assert moved.dtype == np.dtype(np.float64)
    assert moved.shape == (3,)


def test_a_returned_array_owns_its_memory() -> None:
    """Writing into the result cannot reach back into the controller."""
    tracker = _tracker()
    moved = tracker.step(np.array([0.2, 0.0, 0.0]), 0.05)
    moved[0] = 1e9
    assert tracker.q[0] != 1e9


def _radii() -> np.ndarray:
    array = ActuatorArray(actuator_count=6)
    array.init_radii(CircleBody(mass=2.0, radius=1.0))
    return array.radii


def _distances() -> np.ndarray:
    field = KDTreeOccupancy(np.array([[1.0, 1.0], [2.0, 2.0]]), clearance=0.1)
    return field.query_distances(
        np.array([[0.0, 0.0], [1.5, 1.5], [3.0, 3.0]])
    )


def _rates() -> np.ndarray:
    array = ActuatorArray(actuator_count=6)
    array.init_radii(CircleBody(mass=2.0, radius=1.0))
    return array.radii_velocities


def _velocity() -> np.ndarray:
    tracker = _tracker()
    tracker.step(np.array([0.2, 0.0, 0.0]), 0.05)
    return tracker.vel


def test_a_tuple_of_arrays_keeps_one_entry_per_member() -> None:
    """``ReferencePath.sample`` answers with five arrays of one length."""
    sampled = ReferencePath([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]).sample(9)
    assert isinstance(sampled, tuple)
    assert len(sampled) == 5
    for member in sampled:
        assert isinstance(member, np.ndarray)
        assert member.dtype == np.dtype(np.float64)
        assert member.shape == (9,)


@pytest.mark.parametrize(
    ("name", "call", "shape"),
    [
        ("actuator radii", _radii, (6,)),
        ("actuator radial rates", _rates, (6,)),
        ("batched distances", _distances, (3,)),
        ("tracker velocity", _velocity, (3,)),
    ],
)
def test_every_array_returning_entry_point_answers_with_its_shape(
    name: str, call, shape: tuple[int, ...]
) -> None:
    """Shape and dtype survive the trip, whichever module answered."""
    returned = call()
    assert isinstance(returned, np.ndarray), name
    assert returned.dtype == np.dtype(np.float64), name
    assert returned.shape == shape, name


def test_a_nearest_obstacle_answers_with_a_distance_and_a_point() -> None:
    """The pair keeps its shape: a float first, then a planar point."""
    field = KDTreeOccupancy(np.array([[1.0, 1.0], [2.0, 2.0]]), clearance=0.1)
    distance, nearest = field.nearest_obstacle(np.array([0.4, 0.4]))
    assert isinstance(distance, float)
    assert isinstance(nearest, np.ndarray)
    assert nearest.dtype == np.dtype(np.float64)
    assert nearest.shape == (2,)


def test_an_array_argument_is_read_without_being_written_through() -> None:
    """A caller's input survives the call it was passed to."""
    query = np.array([0.4, 0.4])
    untouched = query.copy()
    field = KDTreeOccupancy(np.array([[1.0, 1.0], [2.0, 2.0]]), clearance=0.1)
    field.nearest_obstacle(query)
    assert np.array_equal(query, untouched)
