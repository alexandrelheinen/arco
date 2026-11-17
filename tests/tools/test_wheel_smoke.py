"""Smoke script checks for FR-BUILD-08 (also runs on x86_64)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "wheel_smoke.py"


def test_fr_build_08_wheel_smoke_passes_on_editable_install() -> None:
    """FR-BUILD-08: the shared smoke imports and plans on this host."""
    result = subprocess.run(
        [sys.executable, str(SMOKE)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "imported on" in result.stdout
    assert "planners answered from the installed wheel" in result.stdout
