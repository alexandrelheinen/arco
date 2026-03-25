class DiscretePlanner:
    """
    Base class for discrete (grid-based) planners.
    """

    def __init__(self, grid):
        self.grid = grid
