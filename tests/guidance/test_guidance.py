import numpy as np

from arco.guidance import BSplineInterpolator, DubinsPrimitive


def test_dubins_primitive():
    dubins = DubinsPrimitive(turning_radius=2.0)
    path = dubins.steer((0, 0, 0), (1, 1, 0))
    assert path[0] == (0, 0, 0)
    assert path[-1] == (1, 1, 0)


def test_bspline_interpolator():
    interp = BSplineInterpolator(degree=3)
    discrete_path = [(0, 0), (1, 1), (2, 2)]
    smooth_path = interp.interpolate(discrete_path)
    assert smooth_path == discrete_path
