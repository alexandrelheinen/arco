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

from abc import ABC, abstractmethod


class ExplorationPrimitive(ABC):
    """
    Abstract base for exploration primitives (e.g., Dubins, Reeds-Shepp).
    Used in RRT-based planners to ensure kinematic feasibility.
    """

    @abstractmethod
    def steer(self, from_state, to_state):
        """Return a feasible path segment from from_state to to_state."""
        pass


class DubinsPrimitive(ExplorationPrimitive):
    """
    Dubins path primitive for car-like robots (no reverse, minimum turning radius).
    """

    def __init__(self, turning_radius=1.0):
        self.turning_radius = turning_radius

    def steer(self, from_state, to_state):
        # Placeholder: would use a dubins path library in practice
        return [from_state, to_state]


class Interpolator(ABC):
    """
    Abstract base for interpolation (e.g., B-splines, shortcutting).
    Used to convert discrete node sequences to continuous trajectories.
    """

    @abstractmethod
    def interpolate(self, path):
        """Return a continuous trajectory from a discrete path."""
        pass


class BSplineInterpolator(Interpolator):
    """
    B-spline interpolator for smoothing discrete paths.
    """

    def __init__(self, degree=3):
        self.degree = degree

    def interpolate(self, path):
        # Placeholder: would use scipy.interpolate in practice
        return path
