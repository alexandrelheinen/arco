"""Tests for the release-image shell scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_generate_release_images_dry_run_lists_the_set() -> None:
    """Dry-run prints the renderer command and every release filename."""
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "generate_release_images.sh"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = result.stdout
    assert "tools/render_web.py" in out
    assert "field-nocturne.png" in out
    assert "field-og.png" in out
    assert "mark-atlas.png" in out


def test_publish_release_images_dry_run_requires_every_file(
    tmp_path: Path,
) -> None:
    """A missing plate fails the publish; a complete directory is listed."""
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "publish_release_images.sh"
    listed = subprocess.run(
        ["python3", str(repo / "tools" / "render_web.py"), "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line for line in listed.stdout.splitlines() if line.strip()]
    for name in names:
        (tmp_path / name).write_bytes(b"png")

    ok = subprocess.run(
        ["bash", str(script), "v0.0.0", str(tmp_path), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "would upload" in ok.stdout
    assert "field-og.png" in ok.stdout

    (tmp_path / "field-nocturne.png").unlink()
    missing = subprocess.run(
        ["bash", str(script), "v0.0.0", str(tmp_path), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "field-nocturne.png" in missing.stdout


def test_release_workflow_renders_and_attaches_images() -> None:
    """Publishing a release runs the image job beside the video jobs."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "release.yml"
    )
    text = workflow.read_text()
    assert "generate-images:" in text
    assert "publish-images:" in text
    assert "scripts/generate_release_images.sh" in text
    assert "scripts/publish_release_images.sh" in text
