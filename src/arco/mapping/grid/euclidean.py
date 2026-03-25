import numpy as np

from .base import Grid


class EuclideanGrid(Grid):
    """
    Grid with diagonal (L2) connectivity and distance.
    Includes diagonal neighbors and uses Euclidean distance.
    """

    def __init__(self, shape):
        super().__init__(shape)

    def distance(self, a, b):
        """Return L2 (Euclidean) distance between two nodes."""
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def neighbors(self, idx):
        return super().neighbors(idx, diagonal=True)
