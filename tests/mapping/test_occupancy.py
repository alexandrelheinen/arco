import pytest

from arco.mapping import Occupancy


class DummyOccupancy(Occupancy):
    def is_occupied(self, point):
        return point == (1, 1)


def test_occupancy_abstract():
    with pytest.raises(TypeError):
        Occupancy()


def test_dummy_occupancy():
    occ = DummyOccupancy()
    assert occ.is_occupied((1, 1))
    assert not occ.is_occupied((0, 0))
