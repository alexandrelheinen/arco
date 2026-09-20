"""Still-frame output for the ARCO simulator.

The scenario mains normally pipe every rendered frame to
:class:`~arco.simulator.sim.video.VideoWriter`.  When ``arcosim --still`` is
used the CLI installs a :class:`StillRequest` here, and the mains pick up two
changes through the helpers below: the recording framebuffer is sized from
the request, and the frame sink becomes a :class:`StillSink` that saves the
selected frames as PNG instead of encoding an MP4.

Nothing else about the run changes.  With no active request every helper
returns the previous behaviour, so release recordings are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class CaptureDone(Exception):
    """Raised by :class:`StillSink` once the last frame has been saved.

    The scenario render loops close their writer in a ``finally`` block, so
    raising from the frame sink unwinds the loop cleanly.
    """


class CaptureTooShort(RuntimeError):
    """Raised when a run ended before every requested frame was reached."""


@dataclass(frozen=True)
class StillRequest:
    """A request to save specific recorded frames as PNG files.

    Attributes:
        frames: Zero-based recorded-frame indices to save.
        output: Destination path.  A single frame uses it verbatim; several
            frames add an ``_fNNNNN`` suffix before the extension.
        width: Recording framebuffer width in pixels, or ``None`` to
            keep the scenario's own recording width.
        height: Recording framebuffer height in pixels, or ``None`` to keep
            the scenario's own recording height.
    """

    frames: tuple[int, ...]
    output: Path
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        """Normalize *frames* to a sorted, duplicate-free tuple.

        Raises:
            ValueError: If no frame index was given or any index is
                negative.
        """
        ordered = tuple(sorted({int(f) for f in self.frames}))
        if not ordered:
            raise ValueError("A still request needs at least one frame.")
        if ordered[0] < 0:
            raise ValueError("Frame indices must not be negative.")
        object.__setattr__(self, "frames", ordered)
        object.__setattr__(self, "output", Path(self.output))

    @property
    def last_frame(self) -> int:
        """Highest requested frame index."""
        return self.frames[-1]

    def frame_path(self, frame: int) -> Path:
        """Return the file path for *frame*.

        Args:
            frame: Zero-based recorded-frame index.

        Returns:
            The destination path for that frame.
        """
        if len(self.frames) == 1:
            return self.output
        stem = self.output.stem
        return self.output.with_name(
            f"{stem}_f{frame:05d}{self.output.suffix}"
        )


_request: StillRequest | None = None
_last_sink: "StillSink | None" = None


def set_request(request: StillRequest) -> None:
    """Install the process-level still request.

    Args:
        request: The request the scenario mains should honor.
    """
    global _request, _last_sink
    _request = request
    _last_sink = None


def clear_request() -> None:
    """Remove any installed still request."""
    global _request, _last_sink
    _request = None
    _last_sink = None


def active_request() -> StillRequest | None:
    """Return the installed still request, if any.

    Returns:
        The active :class:`StillRequest`, or ``None`` in normal recording.
    """
    return _request


def resolve_recording_size(
    default_width: int, default_height: int
) -> tuple[int, int]:
    """Return the framebuffer size a recording should use.

    Args:
        default_width: The scenario's own recording width.
        default_height: The scenario's own recording height.

    Returns:
        The requested still size when a request is active, falling back to
        the passed defaults per axis.
    """
    if _request is None:
        return (default_width, default_height)
    return (
        _request.width if _request.width is not None else default_width,
        _request.height if _request.height is not None else default_height,
    )


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    """Write an ``(h, w, 3)`` uint8 array to *path* as an 8-bit RGB PNG.

    Args:
        rgb: Row-major, top-to-bottom RGB pixel array.
        path: Destination file path.  Parent directories are created.
    """
    import pygame  # noqa: PLC0415 — display extra, imported lazily

    path.parent.mkdir(parents=True, exist_ok=True)
    # make_surface expects (w, h, 3); the incoming array is (h, w, 3).
    surface = pygame.surfarray.make_surface(
        np.ascontiguousarray(rgb.transpose(1, 0, 2))
    )
    pygame.image.save(surface, str(path))
    logger.info("Wrote %s (%d x %d)", path, rgb.shape[1], rgb.shape[0])


@dataclass
class StillSink:
    """Frame sink that saves selected frames as PNG.

    The method surface matches
    :class:`~arco.simulator.sim.video.VideoWriter` so a scenario main can
    use either without branching.

    Attributes:
        request: The still request being served.
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.
    """

    request: StillRequest
    width: int
    height: int
    _index: int = field(default=0, init=False)
    _saved: list[int] = field(default_factory=list, init=False)

    # -- lifecycle -----------------------------------------------------
    def open(self) -> None:
        """Start the sink.  There is no encoder subprocess to launch."""
        logger.info(
            "Still capture armed: %d x %d, frames %s",
            self.width,
            self.height,
            list(self.request.frames),
        )

    def close(self) -> None:
        """Stop the sink.  Each still is flushed as it is written."""

    def __enter__(self) -> StillSink:
        """Return self so the sink works as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Do nothing on context exit."""

    # -- reporting -----------------------------------------------------
    @property
    def saved_frames(self) -> tuple[int, ...]:
        """Frame indices written so far."""
        return tuple(self._saved)

    def missing_frames(self) -> tuple[int, ...]:
        """Return requested frames that were never reached.

        Returns:
            The frame indices still outstanding, in ascending order.
        """
        return tuple(f for f in self.request.frames if f not in self._saved)

    # -- frame capture -------------------------------------------------
    def submit(self, rgb: np.ndarray) -> None:
        """Offer one rendered frame to the sink.

        Args:
            rgb: Row-major, top-to-bottom ``(h, w, 3)`` uint8 pixels.

        Raises:
            CaptureDone: Once the last requested frame has been saved.
        """
        if self._index in self.request.frames:
            save_rgb(rgb, self.request.frame_path(self._index))
            self._saved.append(self._index)
        if self._index >= self.request.last_frame:
            raise CaptureDone
        self._index += 1

    def write_frame(self, surface: Any) -> None:
        """Capture a pygame surface frame (scenarios without OpenGL).

        Args:
            surface: The pygame surface that was just presented.
        """
        import pygame  # noqa: PLC0415 — display extra, imported lazily

        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        self.submit(np.ascontiguousarray(rgb))

    def write_frame_gl(self) -> None:
        """Capture the current OpenGL framebuffer.

        Must be called after ``pygame.display.flip()`` while the GL context
        is bound, exactly like
        :meth:`~arco.simulator.sim.video.VideoWriter.write_frame_gl`.
        """
        from OpenGL.GL import (  # type: ignore[import-untyped] # noqa: PLC0415
            GL_RGB,
            GL_UNSIGNED_BYTE,
            glFinish,
            glReadPixels,
        )

        glFinish()
        data = glReadPixels(
            0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE
        )
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        # OpenGL stores rows bottom-to-top; PNG expects top-to-bottom.
        self.submit(np.ascontiguousarray(np.flipud(frame)))


def make_writer(path: str, width: int, height: int, fps: int) -> Any:
    """Return the frame sink a scenario recording should use.

    Args:
        path: Output path passed on the command line.
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.
        fps: Target frame rate, used by the MP4 encoder only.

    Returns:
        A :class:`StillSink` when a still request is active, otherwise a
        :class:`~arco.simulator.sim.video.VideoWriter`.
    """
    global _last_sink
    if _request is not None:
        _last_sink = StillSink(_request, width, height)
        return _last_sink
    from .video import VideoWriter  # noqa: PLC0415 — pulls pygame

    return VideoWriter(path, width, height, fps)


def last_sink() -> "StillSink | None":
    """Return the most recent still sink built by :func:`make_writer`.

    Returns:
        The sink, or ``None`` if no still capture has run in this process.
    """
    return _last_sink


def pin_unseeded_rng(seed: int) -> None:
    """Make unseeded ``numpy`` generators deterministic.

    The scene classes build their planners without passing a seed, so
    ``numpy.random.default_rng()`` draws from OS entropy and every run grows
    a different exploration tree.  Wrapping the factory is the smallest
    change that makes a still reproducible; calls that already pass a seed
    (the city world generator uses ``seed: 42`` from ``map/city.yml``) keep
    it, so the generated world is unaffected.

    Args:
        seed: Seed applied to generators created without one.
    """
    original = np.random.default_rng

    def seeded(s: Any = None, **kwargs: Any) -> Any:
        return original(seed if s is None else s, **kwargs)

    np.random.default_rng = seeded  # type: ignore[assignment]


def parse_frames(spec: str) -> tuple[int, ...]:
    """Parse a ``--still`` value into frame indices.

    Args:
        spec: Comma-separated frame indices, for example ``"300,900"``.

    Returns:
        The parsed indices, sorted and de-duplicated.

    Raises:
        ValueError: If *spec* holds no parsable index.
    """
    return _normalize(int(part) for part in spec.split(",") if part.strip())


def _normalize(frames: Iterable[int]) -> tuple[int, ...]:
    """Sort and de-duplicate *frames*.

    Args:
        frames: Frame indices in any order.

    Returns:
        The indices as a sorted tuple.

    Raises:
        ValueError: If *frames* is empty.
    """
    ordered: Sequence[int] = sorted(set(frames))
    if not ordered:
        raise ValueError("No frame index given.")
    return tuple(ordered)
