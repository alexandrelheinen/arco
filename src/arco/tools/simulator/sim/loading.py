"""Loading-screen overlay rendered during ``scene.build()``.

Runs ``scene.build(progress=reporter)`` in a daemon background thread while
the main (GL) thread pumps events and renders a full-screen loading panel
showing:

* Current build-step name
* Step *x* / *N* counter
* Visual and text percentage bar    (100 % = build finished / race starts)

A :class:`tqdm.tqdm` progress bar is also written to ``stderr``, sitting
alongside the standard logging output used across the simulators.

Usage (call *after* ``pygame.init()`` and ``pygame.display.set_mode()``)::

    from sim.loading import run_with_loading_screen
    run_with_loading_screen(scene, sw, sh)

The scene ``build()`` method must accept an optional ``progress`` keyword
argument and call it at each meaningful milestone::

    def build(self, *, progress=None) -> None:
        if progress is not None:
            progress("Building occupancy map", 1, 3)
        ...

Calling ``progress(step_name, step_index, total_steps)`` at the start of each
step advances both the on-screen panel and the terminal bar.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Callable

import pygame
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glShadeModel,
)

from .. import renderer_gl

try:
    import tqdm as _tqdm_module

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matches the dark-asphalt theme used across scenes)
# ---------------------------------------------------------------------------
_C_BG_DEFAULT: tuple[int, int, int] = (22, 24, 30)
_C_PANEL_BG: tuple[int, int, int, int] = (12, 14, 22, 220)
_C_TITLE: tuple[int, int, int] = (200, 200, 225)
_C_STEP: tuple[int, int, int] = (160, 180, 210)
_C_BAR_EMPTY: tuple[int, int, int] = (35, 40, 55)
_C_BAR_FILL: tuple[int, int, int] = (70, 150, 255)
_C_PCT: tuple[int, int, int] = (140, 165, 200)

# Animation: rotating spinner characters shown when step_index == 0.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Thread-safe progress state updated from the build thread.

    Instances are callable.  The build method calls them as::

        progress("Running RRT*", 2, 4)

    Internally the reporter stores the latest state for the render loop
    and advances a :class:`tqdm.tqdm` bar in the terminal.

    Args:
        title: Short label shown at the top of the loading panel (e.g.
            the scene title).
    """

    def __init__(self, title: str = "") -> None:
        self._title = title
        # Mutable state — written by build thread, read by render thread.
        # Simple Python attribute assignment is atomic under the GIL, so no
        # explicit lock is required for these scalar/string fields.
        self.step_name: str = "Initialising…"
        self.step_index: int = 0
        self.total_steps: int = 1
        self._bar: Any = None  # tqdm.tqdm | None
        self._done: bool = False  # set to True only after build() finishes

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        step_name: str,
        step_index: int,
        total_steps: int,
    ) -> None:
        """Advance progress to *step_index* / *total_steps* with *step_name*.

        Args:
            step_name: Human-readable description of the current build step.
            step_index: 1-based index of the current step.
            total_steps: Total number of build steps.
        """
        self.step_name = step_name
        self.step_index = step_index
        self.total_steps = total_steps
        self._advance_tqdm(step_name, step_index, total_steps)

    # ------------------------------------------------------------------
    # tqdm integration
    # ------------------------------------------------------------------

    def _advance_tqdm(
        self,
        step_name: str,
        step_index: int,
        total_steps: int,
    ) -> None:
        if not _HAS_TQDM:
            logger.info(
                "Loading  [%d/%d]  %s", step_index, total_steps, step_name
            )
            return

        if self._bar is None:
            self._bar = _tqdm_module.tqdm(
                total=total_steps,
                desc=step_name,
                unit="step",
                file=sys.stderr,
                leave=True,
                bar_format=(
                    "{desc:<35} {percentage:3.0f}%|{bar:25}|"
                    " {n_fmt}/{total_fmt} [{elapsed}]"
                ),
                colour="blue",
            )
            self._bar.n = step_index - 1

        self._bar.n = step_index
        self._bar.set_description(f"{step_name:<35}")
        self._bar.refresh()

    def close(self) -> None:
        """Finalize the tqdm bar (call from the build thread when done)."""
        if self._bar is not None:
            self._bar.n = self._bar.total
            self._bar.set_description(f"{'Done':<35}")
            self._bar.refresh()
            self._bar.close()
            self._bar = None
        self._done = True

    # ------------------------------------------------------------------
    # Read-only snapshot (consumed by the render thread)
    # ------------------------------------------------------------------

    @property
    def fraction(self) -> float:
        """Progress fraction in [0, 1]; 1.0 only after build truly finishes.

        While a step is in progress, returns the fraction of *completed*
        steps (i.e. ``step_index - 1`` out of ``total_steps``), so the bar
        never reaches 1.0 until :meth:`close` is called.
        """
        if self._done:
            return 1.0
        t = self.total_steps
        if t <= 0:
            return 0.0
        return (self.step_index - 1) / t


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _make_loading_surface(
    reporter: ProgressReporter,
    sw: int,
    sh: int,
    tick: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> pygame.Surface:
    """Build a pygame SRCALPHA surface for the loading overlay.

    Args:
        reporter: Progress state.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        tick: Frame counter used to animate the spinner.
        title_font: Large bold font for the panel title.
        body_font: Normal font for step name and percentage.

    Returns:
        Full-screen transparent :class:`pygame.Surface`.
    """
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    panel_w = min(520, max(sw - 80, 260))
    panel_h = 170
    px = (sw - panel_w) // 2
    py = (sh - panel_h) // 2

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(_C_PANEL_BG)

    # ── Title ─────────────────────────────────────────────────────────────
    spinner_char = _SPINNER[tick % len(_SPINNER)]
    title_text = f"{spinner_char}  Loading…"
    if reporter._done:
        title_text = "✓  Ready"
    title_surf = title_font.render(title_text, True, _C_TITLE)
    panel.blit(title_surf, ((panel_w - title_surf.get_width()) // 2, 12))

    # ── Step name ─────────────────────────────────────────────────────────
    step_surf = body_font.render(reporter.step_name, True, _C_STEP)
    panel.blit(step_surf, ((panel_w - step_surf.get_width()) // 2, 56))

    # ── Progress bar ──────────────────────────────────────────────────────
    bar_margin = 24
    bar_x = bar_margin
    bar_y = 88
    bar_w = panel_w - 2 * bar_margin
    bar_h = 12
    pygame.draw.rect(
        panel, _C_BAR_EMPTY, (bar_x, bar_y, bar_w, bar_h), border_radius=4
    )
    filled = max(0, int(bar_w * reporter.fraction))
    if filled > 0:
        pygame.draw.rect(
            panel,
            _C_BAR_FILL,
            (bar_x, bar_y, filled, bar_h),
            border_radius=4,
        )
    pygame.draw.rect(
        panel, (60, 70, 90), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4
    )

    # ── Percentage / step counter ─────────────────────────────────────────
    pct_str = (
        f"{reporter.fraction * 100:.0f}%"
        f"  —  step {reporter.step_index} / {reporter.total_steps}"
    )
    pct_surf = body_font.render(pct_str, True, _C_PCT)
    panel.blit(pct_surf, ((panel_w - pct_surf.get_width()) // 2, 110))

    # ── Compact scene title ───────────────────────────────────────────────
    if reporter._title:
        lbl_surf = body_font.render(reporter._title, True, (80, 90, 110))
        panel.blit(lbl_surf, ((panel_w - lbl_surf.get_width()) // 2, 142))

    surf.blit(panel, (px, py))
    return surf


def _render_loading(
    reporter: ProgressReporter,
    sw: int,
    sh: int,
    tick: int,
    bg_color: tuple[int, int, int],
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> None:
    """Clear the framebuffer and blit the loading overlay.

    Args:
        reporter: Current progress state.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        tick: Frame counter for spinner animation.
        bg_color: Background FILL color (RGB 0-255).
        title_font: Large bold font.
        body_font: Normal body font.
    """
    r, g, b = bg_color[0] / 255.0, bg_color[1] / 255.0, bg_color[2] / 255.0
    glClearColor(r, g, b, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glShadeModel(GL_SMOOTH)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    overlay = _make_loading_surface(
        reporter, sw, sh, tick, title_font, body_font
    )
    renderer_gl.blit_overlay(overlay, 0, 0, sw, sh)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_with_loading_screen(
    scene: Any,
    sw: int,
    sh: int,
    *,
    fps: int = 30,
    bg_color: tuple[int, int, int] = _C_BG_DEFAULT,
) -> None:
    """Run ``scene.build(progress=reporter)`` with a full-screen loading overlay.

    The build runs in a daemon thread.  The caller's (main) thread renders
    the loading panel at *fps* frames per second while waiting.  Any
    exception raised inside ``build()`` is re-raised in the main thread after
    the thread joins.

    This function is a drop-in replacement for a bare ``scene.build()`` call.
    The pygame window and OpenGL context must already be created before
    calling this function.

    Args:
        scene: Scene object whose ``build(progress=...)`` will be called.
        sw: Screen width in pixels (for overlay sizing).
        sh: Screen height in pixels (for overlay sizing).
        fps: Target overlay refresh rate in frames per second.
        bg_color: Background RGB fill color (should match the scene's
            ``bg_color`` property).

    Raises:
        Exception: Any exception raised by ``scene.build()`` is re-raised
            here after the loading screen exits.
    """
    pygame.font.init()
    title_font = pygame.font.SysFont("monospace", 26, bold=True)
    body_font = pygame.font.SysFont("monospace", 14)

    title = getattr(scene, "title", "") if hasattr(scene, "title") else ""
    # ``title`` may be an abstract property that raises before build() —
    # guard against attribute errors.
    try:
        title = str(title)
    except Exception:
        title = ""

    reporter = ProgressReporter(title=title)
    error: list[BaseException] = []
    clock = pygame.time.Clock()

    def _build() -> None:
        try:
            scene.build(progress=reporter)
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            reporter.close()

    thread = threading.Thread(target=_build, daemon=True)
    thread.start()
    logger.info("Build thread started — loading screen active.")

    tick = 0
    while thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        _render_loading(
            reporter, sw, sh, tick, bg_color, title_font, body_font
        )
        pygame.display.flip()
        clock.tick(fps)
        tick += 1

    thread.join()

    if error:
        raise error[0]

    logger.info("Build complete — starting simulation.")
