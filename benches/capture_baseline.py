"""Capture the public API signatures baseline for parity testing.
The snapshot recorded here is verified on every test run by
``tests/rust/test_signature_parity.py``.

Usage:
    python benches/capture_baseline.py
"""

from __future__ import annotations

import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

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


def main() -> int:
    """Capture public API signatures and write under ``benches/baseline``."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    environment = _environment()

    payload = capture_signatures()
    payload["environment"] = environment
    target = BASELINE_DIR / "signatures.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target.relative_to(Path.cwd())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
