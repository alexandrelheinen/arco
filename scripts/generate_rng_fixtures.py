"""Capture numpy PCG64 reference vectors for the Rust implementation.

FR-RNG-02 requires that a seeded ARCO planner draw the same values the
Python implementation drew, which means reproducing both halves of what
``numpy.random.default_rng`` does: the SeedSequence that turns a seed into
a 128 bit state and increment, and the PCG64 output function that turns
that state into draws.

The fixtures this writes are the oracle for
``crates/arco-core/tests/pcg64.rs``.  Regenerate them only when numpy's
own behavior changes, which would itself be a breaking change worth a line
in ``docs/decisions.md``.

Usage:
    python scripts/generate_rng_fixtures.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "arco-core"
    / "tests"
    / "fixtures"
    / "pcg64.json"
)

# Scalar seeds covering zero, small values, the seeds the ARCO benchmarks
# already use, and a value past 64 bits so that the SeedSequence entropy
# path is exercised rather than only its fast case.
SEEDS = [0, 1, 7, 42, 12345, 20260916, 2**64 + 1]

RAW_COUNT = 16
DOUBLE_COUNT = 8


def _case(seed: int) -> dict[str, object]:
    """Capture the derived state and the first draws for one seed."""
    state_only = np.random.default_rng(seed).bit_generator.state["state"]

    raw_generator = np.random.default_rng(seed).bit_generator
    raw = [int(value) for value in raw_generator.random_raw(RAW_COUNT)]

    doubles = [float(value) for value in np.random.default_rng(seed).random(DOUBLE_COUNT)]

    return {
        "seed": str(seed),
        "initial_state": f"{state_only['state']:#034x}",
        "increment": f"{state_only['inc']:#034x}",
        "raw_u64": [f"{value:#018x}" for value in raw],
        # The bit patterns are what the test compares.  A decimal literal
        # would make the assertion depend on two float parsers agreeing,
        # and serde_json's differs from Rust's own by one unit in the last
        # place on at least one of these values.
        "random_f64_bits": [
            f"{struct.unpack('<Q', struct.pack('<d', value))[0]:#018x}" for value in doubles
        ],
        # Kept for a human reading the fixture, never asserted against.
        "random_f64": doubles,
    }


def main() -> int:
    """Write the fixture file and report where it went."""
    payload = {
        "source": "numpy.random.default_rng, PCG64 with the XSL-RR output function",
        "numpy_version": np.__version__,
        "raw_count": RAW_COUNT,
        "double_count": DOUBLE_COUNT,
        "cases": [_case(seed) for seed in SEEDS],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(Path.cwd())} with {len(SEEDS)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
