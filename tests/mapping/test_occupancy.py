import numpy as np
import pytest

from arco.mapping import Occupancy


class DummyOccupancy(Occupancy):
    def nearest_obstacle(self, point):
        obstacle = np.array([1.0, 1.0])
        dist = float(np.linalg.norm(np.asarray(point) - obstacle))
        return dist, obstacle

    def is_occupied(self, point):
        dist, _ = self.nearest_obstacle(point)
        return dist < 0.5


def test_occupancy_abstract():
    with pytest.raises(TypeError):
        Occupancy()


def test_dummy_occupancy():
    occ = DummyOccupancy()
    assert occ.is_occupied(np.array([1.0, 1.0]))
    assert not occ.is_occupied(np.array([0.0, 0.0]))
