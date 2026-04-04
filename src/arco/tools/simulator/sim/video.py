"""ffmpeg-backed video writer for the ARCO simulator.

:class:`VideoWriter` is a context manager that captures raw Pygame frames
and pipes them to an ``ffmpeg`` subprocess producing an MP4 file.
"""

from __future__ import annotations

import logging
import subprocess
from types import TracebackType

import numpy as np
import pygame

logger = logging.getLogger(__name__)


class VideoWriter:
    """Context manager that pipes raw RGB frames to an ffmpeg MP4 encoder.

    Args:
        path: Output MP4 file path.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._proc: subprocess.Popen[bytes] | None = None

    def open(self) -> None:
        """Launch the ffmpeg subprocess.

        Raises:
            RuntimeError: If the writer is already open.
        """
        if self._proc is not None:
            raise RuntimeError("VideoWriter is already open.")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            self._path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(
            "Recording to %r (%dx%d @ %d fps)",
            self._path,
            self._width,
            self._height,
            self._fps,
        )

    def write_frame(self, surface: pygame.Surface) -> None:
        """Capture *surface* and write it to the ffmpeg pipe.

        Args:
            surface: Pygame surface to capture.

        Raises:
            RuntimeError: If the writer has not been opened.
        """
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("VideoWriter is not open.")
        frame = pygame.surfarray.array3d(surface)
        frame = np.ascontiguousarray(frame.transpose(1, 0, 2))
        self._proc.stdin.write(frame.tobytes())

    def write_frame_gl(self) -> None:
        """Capture the current OpenGL framebuffer and write to ffmpeg.

        Reads raw RGB pixels via ``glReadPixels``, flips vertically
        (OpenGL origin is bottom-left; video codecs expect top-left),
        then writes the row-major RGB bytes to the ffmpeg stdin pipe.

        Must be called **after** ``pygame.display.flip()`` and while an
        active OpenGL context is bound to the current thread.

        Raises:
            RuntimeError: If the writer has not been opened.
        """
        from OpenGL.GL import (  # type: ignore[import-untyped]
            GL_RGB,
            GL_UNSIGNED_BYTE,
            glFinish,
            glReadPixels,
        )

        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("VideoWriter is not open.")
        glFinish()
        data = glReadPixels(
            0, 0, self._width, self._height, GL_RGB, GL_UNSIGNED_BYTE
        )
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self._height, self._width, 3
        )
        # OpenGL stores rows bottom-to-top; ffmpeg expects top-to-bottom.
        self._proc.stdin.write(
            np.ascontiguousarray(np.flipud(frame)).tobytes()
        )

    def close(self) -> None:
        """Close the ffmpeg pipe and wait for the subprocess to finish."""
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()
        returncode = self._proc.wait()
        self._proc = None
        if returncode != 0:
            logger.error(
                "ffmpeg exited with code %d; video may be incomplete.",
                returncode,
            )
        else:
            logger.info("Video saved to %r.", self._path)

    def __enter__(self) -> VideoWriter:
        """Open the writer on context entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the writer on context exit."""
        self.close()
