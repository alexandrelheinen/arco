"""Capture the pre-port Python baselines the Rust port is measured against.

Run this before any Rust implementation exists.  Capturing a baseline
after the fact lets the target drift to whatever the port achieved, which
is why FR-PERF-01, FR-MPC-05 and FR-API-02 all name a recorded baseline
rather than a relative improvement.

Usage:
    python benches/capture_baseline.py [--quick]
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"


def _environment() -> dict[str, Any]:
    """Record enough of the machine to make a comparison honest."""
    import numpy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": numpy.__version__,
    }


def _time_calls(fn: Callable[[], Any], repeat: int) -> dict[str, float]:
    """Run ``fn`` ``repeat`` times and summarize the wall-clock durations."""
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    durations.sort()
    return {
        "repeat": repeat,
        "min_seconds": durations[0],
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.fmean(durations),
        "max_seconds": durations[-1],
    }


# --------------------------------------------------------------------------
# FR-API-02: public signature snapshot
# --------------------------------------------------------------------------


def _public_modules() -> list[str]:
    """Every ``arco`` package that exports a public name, simulator aside."""
    import pkgutil

    import arco

    modules = []
    for info in pkgutil.walk_packages(arco.__path__, prefix="arco."):
        if ".simulator" in info.name or info.name.endswith(".simulator"):
            continue
        modules.append(info.name)
    return ["arco"] + sorted(modules)


def _has_python_init(obj: type) -> bool:
    """Whether the nearest `__init__` above *obj* is written in Python.

    A compiled class may carry an `__init__` that exists only to absorb a
    subclass calling `super().__init__(...)`, since construction happened
    in `__new__`.  Its signature is `(*args, **kwargs)` and describes
    nothing, so the real constructor signature has to come off the class
    instead.  Only a Python `__init__` is worth reading directly.
    """
    for base in obj.__mro__:
        if base is object:
            continue
        found = vars(base).get("__init__")
        if found is not None:
            return inspect.isfunction(found)
    return False


def _class_members(obj: type) -> list[str]:
    """Return the public member names a caller can reach on *obj*.

    Resolved through the class rather than read out of ``vars``, because a
    compiled class does not hold what it inherits in its own dictionary:
    ``ManhattanGrid.neighbors`` comes from ``Grid`` and
    ``vars(ManhattanGrid)`` does not mention it, while
    ``ManhattanGrid().neighbors`` works exactly as it always did.  Reading
    the dictionary reports such a name as vanished when nothing about the
    caller's access to it changed.

    ``__init__`` is treated as one construction contract with ``__new__``
    for the same reason: a class built by PyO3 carries ``__new__`` and no
    ``__init__`` of its own, and ``Class(...)`` is unaffected.
    """
    reachable = {
        name
        for name in dir(obj)
        if not name.startswith("_") and callable(getattr(obj, name, None))
    }
    if callable(getattr(obj, "__init__", None)) or callable(getattr(obj, "__new__", None)):
        reachable.add("__init__")
    return sorted(reachable)


def capture_signatures() -> dict[str, Any]:
    """Snapshot every public callable's signature, keyed by import path.

    FR-API-02 requires that a caller's existing call sites keep working.
    The snapshot taken here is what the parity test compares against once
    the Rust implementation is in place.
    """
    import importlib

    entries: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for module_name in _public_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            skipped[module_name] = f"{type(exc).__name__}: {exc}"
            continue

        exported = getattr(module, "__all__", None)
        if not exported:
            continue

        for name in exported:
            obj = getattr(module, name, None)
            if obj is None:
                skipped[f"{module_name}.{name}"] = "missing from module"
                continue
            path = f"{module_name}.{name}"
            if inspect.isclass(obj):
                for attr in _class_members(obj):
                    member = getattr(obj, attr, None)
                    if attr == "__init__" and not _has_python_init(obj):
                        # A compiled class carries its constructor
                        # signature on the class itself, through
                        # `__text_signature__`, rather than on a
                        # `__init__` it does not define.
                        member = obj
                    if not callable(member):
                        continue
                    try:
                        entries[f"{path}.{attr}"] = str(inspect.signature(member))
                    except (TypeError, ValueError) as exc:
                        skipped[f"{path}.{attr}"] = str(exc)
            elif callable(obj):
                try:
                    entries[path] = str(inspect.signature(obj))
                except (TypeError, ValueError) as exc:
                    skipped[path] = str(exc)

    return {
        "requirement": "FR-API-02",
        "signature_count": len(entries),
        "signatures": dict(sorted(entries.items())),
        "skipped": dict(sorted(skipped.items())),
    }


# --------------------------------------------------------------------------
# FR-PERF-01: RRT* planning throughput
# --------------------------------------------------------------------------

RRT_SCENARIO = {
    "bounds": [(0.0, 50.0), (0.0, 50.0)],
    "start": [2.0, 2.0],
    "goal": [48.0, 48.0],
    "obstacle_count": 400,
    "clearance": 1.2,
    "max_sample_count": 4000,
    "step_size": 2.0,
    "goal_tolerance": 1.5,
    "map_seed": 20260916,
    "planner_seed": 7,
}


def _rrt_occupancy() -> Any:
    """Build the fixed obstacle field the benchmark plans through."""
    from arco.mapping import KDTreeOccupancy

    rng = np.random.default_rng(RRT_SCENARIO["map_seed"])
    points = rng.uniform(4.0, 46.0, size=(RRT_SCENARIO["obstacle_count"], 2))
    return KDTreeOccupancy(points, clearance=RRT_SCENARIO["clearance"])


def capture_rrt(repeat: int) -> dict[str, Any]:
    """Time ``RRTPlanner.plan`` on a fixed cluttered scenario.

    All policy hooks stay at their defaults, which is the condition
    FR-PERF-01 states: the ten-times target applies to the native dispatch
    path, not to a planner driven by an injected Python callable.
    """
    from arco.planning import RRTPlanner

    occupancy = _rrt_occupancy()
    start = np.asarray(RRT_SCENARIO["start"], dtype=float)
    goal = np.asarray(RRT_SCENARIO["goal"], dtype=float)
    found = {"count": 0, "length": None}

    def run() -> None:
        planner = RRTPlanner(
            occupancy,
            RRT_SCENARIO["bounds"],
            max_sample_count=RRT_SCENARIO["max_sample_count"],
            step_size=RRT_SCENARIO["step_size"],
            goal_tolerance=RRT_SCENARIO["goal_tolerance"],
            seed=RRT_SCENARIO["planner_seed"],
        )
        path = planner.plan(start, goal)
        if path is not None:
            found["count"] += 1
            found["length"] = len(path)

    timing = _time_calls(run, repeat)
    return {
        "requirement": "FR-PERF-01",
        "scenario": RRT_SCENARIO,
        "solutions_found": found["count"],
        "path_state_count": found["length"],
        "timing": timing,
        "target": "Rust completes at least 10x faster on the same machine",
    }


# --------------------------------------------------------------------------
# FR-MPC-05: path-following MPC solve time
# --------------------------------------------------------------------------

MPC_SCENARIO = {
    "waypoint_count": 40,
    "amplitude": 3.0,
    "wavelength": 20.0,
    "spacing": 1.0,
    "dt": 0.05,
    "start_pose": [0.0, 1.0, 0.0],
    "speed": 0.3,
    "turn_rate": 0.0,
}


def capture_mpc(repeat: int) -> dict[str, Any]:
    """Time one path-following MPC control step.

    FR-MPC-05 bounds the reformulated controller's median solve time by
    the CasADi implementation's median on the same input, so the median is
    the figure that matters here, not the mean.
    """
    from arco.control.mpc import (
        DubinsPathFollowingMPC,
        DubinsVehicleLimits,
        PathFollowingMPCConfig,
    )

    spacing = MPC_SCENARIO["spacing"]
    amplitude = MPC_SCENARIO["amplitude"]
    wavelength = MPC_SCENARIO["wavelength"]
    waypoints = [
        (i * spacing, amplitude * float(np.sin(2.0 * np.pi * i * spacing / wavelength)))
        for i in range(MPC_SCENARIO["waypoint_count"])
    ]

    limits = DubinsVehicleLimits(
        max_speed=0.5,
        min_speed=0.0,
        max_turn_rate=1.0,
        max_acceleration=0.5,
        max_turn_rate_dot=2.0,
    )
    config = PathFollowingMPCConfig(dt=MPC_SCENARIO["dt"])
    controller = DubinsPathFollowingMPC(vehicle_limits=limits, config=config)
    controller.set_reference(waypoints)

    pose = tuple(MPC_SCENARIO["start_pose"])
    outcome = {"solver_success": None, "solver_status": None}

    # One untimed call so that CasADi's code generation and the first
    # IPOPT setup do not land inside the measured samples.
    warm = controller.step(
        pose,
        speed=MPC_SCENARIO["speed"],
        turn_rate=MPC_SCENARIO["turn_rate"],
        dt=MPC_SCENARIO["dt"],
    )
    outcome["solver_success"] = bool(warm.solver_success)
    outcome["solver_status"] = str(warm.solver_status)

    def run() -> None:
        controller.step(
            pose,
            speed=MPC_SCENARIO["speed"],
            turn_rate=MPC_SCENARIO["turn_rate"],
            dt=MPC_SCENARIO["dt"],
        )

    timing = _time_calls(run, repeat)
    return {
        "requirement": "FR-MPC-05",
        "scenario": MPC_SCENARIO,
        "first_step": outcome,
        "timing": timing,
        "target": "Rust median solve time does not exceed this median",
    }


def main() -> int:
    """Capture every baseline and write it under ``benches/baseline``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fewer repetitions, for checking the script rather than recording",
    )
    args = parser.parse_args()

    rrt_repeat = 3 if args.quick else 15
    mpc_repeat = 10 if args.quick else 200

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    environment = _environment()

    captures = {
        "signatures.json": capture_signatures(),
        "rrt_star.json": capture_rrt(rrt_repeat),
        "mpc_path_following.json": capture_mpc(mpc_repeat),
    }

    for filename, payload in captures.items():
        payload["environment"] = environment
        target = BASELINE_DIR / filename
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {target.relative_to(Path.cwd())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
