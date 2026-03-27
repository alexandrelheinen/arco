# ARCO Tool Scripts

This folder contains utility scripts for visualizing and experimenting with ARCO's planning and mapping modules.

## plot_astar_grid.py

Visualizes a 2D Manhattan grid and computes the shortest A* path from one corner to the opposite.

### Parameters
- **length**: Side length of the grid in meters (default: 100.0)
- **resolution**: Size of each grid cell in meters (default: 0.5)

You can edit these parameters at the top of the script.


### Setup: Create a Virtual Environment and Install Dependencies

It is recommended to use a Python virtual environment for development and running tools.

1. Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install ARCO with all development dependencies from the local source:

```bash
pip install '.[dev]'
```

This will install ARCO and all required packages (including matplotlib, numpy, pytest, black, isort, etc.).

### How to Run

Run the script from the project root:

```bash
python ./tools/plot_astar_grid.py
```

A window will appear showing the empty grid and the A* path (a straight line from start to goal).

- The grid is shown in grayscale.
- The A* path is shown in red.
- Start and goal are marked in green and blue, respectively.

### Next Steps
- In future versions, you will be able to draw obstacles interactively.
- You can modify the script to experiment with different grid sizes, resolutions, or planner settings.
