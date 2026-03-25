import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
import numpy as np

from arco.planning import AStar, DStarLite


def test_planning_api_imports():
    grid = np.zeros((3, 3), dtype=int)
    astar = AStar(grid)
    dstar = DStarLite(grid)
    assert hasattr(astar, "search")
    assert hasattr(dstar, "search")
