from .base import Grid


class ManhattanGrid(Grid):
    """
    Grid with Manhattan (L1) connectivity and distance.
    Only axis-aligned neighbors are considered.
    """

    def __init__(self, shape):
        super().__init__(shape)

    def distance(self, a, b):
        """Return L1 (Manhattan) distance between two nodes."""
        return sum(abs(x - y) for x, y in zip(a, b))

    def neighbors(self, idx):
        return super().neighbors(idx, diagonal=False)
