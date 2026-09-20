"""Tests for the arcosim still-frame sink.

These tests must not import pygame or PyOpenGL: CI unit jobs install
``arco`` without the display extras, so :mod:`arco.simulator.sim.still`
imports both lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arco.simulator.sim import still


@pytest.fixture(autouse=True)
def _clear_request() -> None:
    """Leave no still request behind between tests."""
    still.clear_request()
    yield
    still.clear_request()


def _frame(width: int = 4, height: int = 3) -> np.ndarray:
    """Return a deterministic ``(h, w, 3)`` uint8 test frame."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(
        height, width, 3
    )


# AC-STILL-04
def test_recording_size_falls_back_to_scenario_defaults() -> None:
    assert still.resolve_recording_size(1280, 720) == (1280, 720)


# AC-STILL-03
def test_recording_size_uses_requested_thumbnail_size(
    tmp_path: Path,
) -> None:
    still.set_request(
        still.StillRequest(
            frames=(10,),
            output=tmp_path / "x.png",
            width=1920,
            height=1080,
        )
    )
    assert still.resolve_recording_size(1280, 720) == (1920, 1080)


# AC-STILL-01
def test_single_frame_keeps_the_requested_output_name(
    tmp_path: Path,
) -> None:
    request = still.StillRequest(
        frames=(7,), output=tmp_path / "city.png", width=8, height=8
    )
    assert request.frame_path(7) == tmp_path / "city.png"


# AC-STILL-02
def test_multiple_frames_are_suffixed_with_the_frame_index(
    tmp_path: Path,
) -> None:
    request = still.StillRequest(
        frames=(7, 9), output=tmp_path / "city.png", width=8, height=8
    )
    assert request.frame_path(7) == tmp_path / "city_f00007.png"
    assert request.frame_path(9) == tmp_path / "city_f00009.png"


# AC-STILL-01
def test_sink_saves_only_the_requested_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[tuple[int, Path]] = []
    monkeypatch.setattr(
        still,
        "save_rgb",
        lambda rgb, path: saved.append((int(rgb[0, 0, 0]), path)),
    )
    request = still.StillRequest(
        frames=(1,), output=tmp_path / "o.png", width=4, height=3
    )
    still.set_request(request)
    sink = still.make_writer("ignored", 4, 3, 30)

    sink.submit(_frame())  # frame 0 — not requested
    with pytest.raises(still.CaptureDone):
        sink.submit(_frame())  # frame 1 — requested, and the last one

    assert [p for _, p in saved] == [tmp_path / "o.png"]
    assert sink.saved_frames == (1,)


# AC-STILL-05
def test_missing_frames_are_reported_when_the_run_ends_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(still, "save_rgb", lambda rgb, path: None)
    request = still.StillRequest(
        frames=(0, 5), output=tmp_path / "o.png", width=4, height=3
    )
    still.set_request(request)
    sink = still.make_writer("ignored", 4, 3, 30)
    sink.submit(_frame())

    assert sink.missing_frames() == (5,)


# AC-STILL-04
def test_writer_factory_returns_the_video_writer_without_a_request() -> None:
    pytest.importorskip("pygame")
    from arco.simulator.sim.video import VideoWriter

    writer = still.make_writer("/tmp/out.mp4", 1280, 720, 30)
    assert isinstance(writer, VideoWriter)


# AC-STILL-06
def test_seed_pin_leaves_explicitly_seeded_generators_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", original)
    still.pin_unseeded_rng(1234)
    try:
        pinned = np.random.default_rng().random(3)
        pinned_again = np.random.default_rng().random(3)
        explicit = np.random.default_rng(99).random(3)
    finally:
        np.random.default_rng = original
    assert np.array_equal(pinned, pinned_again)
    assert np.array_equal(explicit, original(99).random(3))
