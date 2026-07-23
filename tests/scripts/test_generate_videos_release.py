"""Tests for scripts/generate_videos.sh release-mode remapping."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_generate_videos_release_dry_run_remaps_city() -> None:
    """Release mode must use city_mpc_preview.yml and pass --fast-record."""
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "generate_videos.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--release",
            "--only",
            "city",
            "--dry-run",
            "--duration",
            "60",
            "--out-dir",
            "/tmp/arco_videos_test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = result.stdout
    assert "city_mpc_preview.yml" in out
    assert "arcosim_city.mp4" in out
    assert "--fast-record" in out
    assert "Mode             : release" in out
