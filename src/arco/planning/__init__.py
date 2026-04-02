# Copyright 2026 alexandre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .discrete import AStarPlanner, DiscretePlanner


class AStar:
    """
    API wrapper for A* planner compatible with test suite.
    Accepts a numpy grid (0=free, 1=occupied).
    By default uses ManhattanGrid; pass grid_type='euclidean' for EuclideanGrid.
    """

    def __init__(self, grid, grid_type="manhattan"):
        from arco.mapping import EuclideanGrid, ManhattanGrid

        if grid_type == "euclidean":
            self.grid = EuclideanGrid(grid.shape)
        else:
            self.grid = ManhattanGrid(grid.shape)
        self.grid.data = np.array(grid, dtype=np.uint8)
        from .planner import AStarPlanner

        self._planner = AStarPlanner(self.grid)

    def search(self, start, goal):
        return self._planner.plan(start, goal)


class DStarLite:
    """
    API wrapper for D* planner (stub — not yet implemented).
    """

    def __init__(self, grid):
        pass

    def search(self, start, goal):
        raise NotImplementedError("D* planner not yet implemented.")
