"""Public API parity against the pre-port snapshot.

Guards FR-API-01 and FR-API-02: every public name resolves from the module
path it always did, and every public callable keeps the argument names,
positional order and default values a caller already relies on.

The snapshot lives in ``benches/baseline/signatures.json`` and was taken
from the pure-Python implementation before any Rust existed.  Regenerate
it only to record a deliberate, reviewed API change, never to make this
test pass.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "benches" / "baseline" / "signatures.json"

# Capturing the current signatures imports every arco package, which
# populates the module-level config caches.  Doing that inside the test
# session leaks into whatever runs next, so the capture happens in its own
# interpreter and only the JSON crosses back.
_CAPTURE = (
    "import json, sys; sys.path.insert(0, %r); "
    "import capture_baseline; "
    "print(json.dumps(capture_baseline.capture_signatures()['signatures']))"
)


@pytest.fixture(scope="module")
def snapshot() -> dict[str, str]:
    """The recorded pre-port signatures, keyed by dotted import path."""
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE}; run benches/capture_baseline.py")
    return json.loads(BASELINE.read_text())["signatures"]


@pytest.fixture(scope="module")
def current() -> dict[str, str]:
    """The signatures the installed package exposes right now."""
    completed = subprocess.run(
        [sys.executable, "-c", _CAPTURE % str(REPO_ROOT / "benches")],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        pytest.fail(f"signature capture failed:\n{completed.stderr}")
    return json.loads(completed.stdout.splitlines()[-1])


def test_no_public_name_disappears(snapshot, current):
    """FR-API-01: every recorded public name still resolves."""
    missing = sorted(set(snapshot) - set(current))
    assert not missing, f"{len(missing)} public names vanished: {missing[:10]}"


#: Names whose signature a recorded deviation says will differ.
#:
#: A-23: ``Grid`` is abstract for real in the port, so it cannot be
#: constructed and the signature of its constructor describes nothing a
#: caller can reach.
_EXEMPT = frozenset(
    {
        "arco.mapping.Grid.__init__",
        "arco.mapping.grid.Grid.__init__",
    }
)

_ANNOTATION = re.compile(r":\s*(\"[^\"]*\"|'[^']*')")
_RETURN = re.compile(r"\s*->\s*.*$")


def _comparable(signature: str) -> str:
    """Strip what deviation A-27 says a caller cannot observe.

    Three things differ between a Python signature and the one a compiled
    class reports, and none of them change how an existing call site
    behaves: the type annotations, the return annotation, and the ``/``
    marking the arguments before it as positional only.  Argument names,
    their order and their defaults are what `FR-API-02` is about and they
    survive this normalization untouched, so a real change still fails.
    """
    stripped = _RETURN.sub("", _ANNOTATION.sub("", signature))
    return re.sub(r"\s+", "", stripped.replace(", /", "").replace("(self,", "(").replace("(self)", "()"))


def test_no_public_signature_changes(snapshot, current):
    """FR-API-02: every recorded signature is unchanged where it counts."""
    changed = {
        name: (snapshot[name], current[name])
        for name in sorted(set(snapshot) & set(current))
        if name not in _EXEMPT
        and _comparable(snapshot[name]) != _comparable(current[name])
    }
    assert not changed, "\n".join(
        f"{name}\n  was: {was}\n  now: {now}" for name, (was, now) in changed.items()
    )
