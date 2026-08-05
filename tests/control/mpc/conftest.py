"""Shared fixtures for path-following MPC tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from arco.mapping.occupancy import Occupancy


class RectOccupancy(Occupancy):
    """Axis-aligned rectangular obstacle occupancy for synthetic tests."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        clearance: float = 0.5,
    ) -> None:
        super().__init__()
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Rectangle bounds must be non-empty.")
        if clearance <= 0.0:
            raise ValueError("clearance must be positive.")
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.clearance = float(clearance)

    def nearest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        pt = np.asarray(point, dtype=float).reshape(-1)
        qx = float(np.clip(pt[0], self.x_min, self.x_max))
        qy = float(np.clip(pt[1], self.y_min, self.y_max))
        # If the query is inside the rectangle, nearest surface point.
        inside = (
            self.x_min <= pt[0] <= self.x_max
            and self.y_min <= pt[1] <= self.y_max
        )
        if inside:
            dist_left = pt[0] - self.x_min
            dist_right = self.x_max - pt[0]
            dist_bottom = pt[1] - self.y_min
            dist_top = self.y_max - pt[1]
            side = int(
                np.argmin([dist_left, dist_right, dist_bottom, dist_top])
            )
            if side == 0:
                nearest = np.array([self.x_min, pt[1]], dtype=float)
            elif side == 1:
                nearest = np.array([self.x_max, pt[1]], dtype=float)
            elif side == 2:
                nearest = np.array([pt[0], self.y_min], dtype=float)
            else:
                nearest = np.array([pt[0], self.y_max], dtype=float)
            dist = float(np.linalg.norm(pt[:2] - nearest))
            return dist, nearest
        nearest = np.array([qx, qy], dtype=float)
        dist = float(np.linalg.norm(pt[:2] - nearest))
        return dist, nearest

    def is_occupied(self, point: np.ndarray) -> bool:
        dist, _ = self.nearest_obstacle(point)
        return dist < self.clearance


@pytest.fixture
def straight_path() -> list[tuple[float, float]]:
    return [(float(i), 0.0) for i in range(0, 31)]
