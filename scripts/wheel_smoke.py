#!/usr/bin/env python3
"""Import and planning smoke for an installed ARCO wheel.

Shared by ``.github/workflows/wheels.yml`` and
``scripts/verify_wheel_aarch64.sh`` so CI and the local aarch64 path
exercise the same definition of "the wheel works" (FR-BUILD-01/08).
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    """Import every layer and solve the A* and RRT smoke problems."""
    import arco
    from arco import (  # noqa: F401
        control,
        guidance,
        kinematics,
        mapping,
        middleware,
        pipeline,
        planning,
        protocols,
    )
    from arco._arco import version

    print(
        "arco",
        version(),
        "imported on",
        sys.version.split()[0],
        "arch",
        platform.machine(),
    )

    import numpy as np
    from arco.mapping import KDTreeOccupancy
    from arco.planning import AStar, RRTPlanner

    grid = np.zeros((20, 20), dtype=np.uint8)
    assert AStar(grid, grid_type="euclidean").search((0, 0), (19, 19))

    occupancy = KDTreeOccupancy(np.array([[5.0, 5.0]]), clearance=0.5)
    planner = RRTPlanner(
        occupancy,
        [(0.0, 10.0), (0.0, 10.0)],
        max_sample_count=500,
        seed=7,
    )
    assert planner.plan(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    print("planners answered from the installed wheel")
    # Keep ``arco`` referenced so an unused-import lint does not strip it.
    _ = arco
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
