"""Camera helpers for the ARCO simulator.

Provides :class:`FollowTransform` (a world-to-screen transform centred on an
arbitrary position) and :class:`CameraFilter` (a critically-damped second-order
spring that smoothly chases a moving vehicle).
"""

from __future__ import annotations


class FollowTransform:
    """World-to-screen transform centred on a filtered camera position.

    Args:
        center_x: Camera centre x in world metres.
        center_y: Camera centre y in world metres.
        screen_size: ``(width, height)`` of the display in pixels.
        scale: Pixels per metre.
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        screen_size: tuple[int, int],
        scale: float,
    ) -> None:
        self._center_x = center_x
        self._center_y = center_y
        self._screen_w, self._screen_h = screen_size
        self._scale = scale

    def __call__(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates to screen pixels.

        Args:
            wx: World x in metres.
            wy: World y in metres.

        Returns:
            Integer ``(screen_x, screen_y)``.
        """
        sx = int(self._screen_w / 2 + (wx - self._center_x) * self._scale)
        sy = int(self._screen_h / 2 - (wy - self._center_y) * self._scale)
        return (sx, sy)

    @property
    def scale(self) -> float:
        """Pixels per metre."""
        return self._scale


class CameraFilter:
    """Second-order linear filter that smoothly chases a moving target.

    Implements a critically-damped spring with configurable natural frequency.

    Args:
        x: Initial camera x position in world metres.
        y: Initial camera y position in world metres.
        natural_frequency: Natural frequency ω₀ in rad/s (default 3.0).
        damping_ratio: Damping ratio ζ (default 1.0 = critically damped).
    """

    def __init__(
        self,
        x: float,
        y: float,
        natural_frequency: float = 3.0,
        damping_ratio: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self._vx = 0.0
        self._vy = 0.0
        self._wn = natural_frequency
        self._zeta = damping_ratio

    def reset(self, x: float, y: float) -> None:
        """Snap the camera to ``(x, y)`` and clear velocity.

        Args:
            x: New camera x position in world metres.
            y: New camera y position in world metres.
        """
        self.x = x
        self.y = y
        self._vx = 0.0
        self._vy = 0.0

    def update(self, target_x: float, target_y: float, dt: float) -> None:
        """Advance the camera one timestep toward the target position.

        Args:
            target_x: Target x position in world metres.
            target_y: Target y position in world metres.
            dt: Timestep in seconds.
        """
        wn, zeta = self._wn, self._zeta
        ax = wn * wn * (target_x - self.x) - 2.0 * zeta * wn * self._vx
        ay = wn * wn * (target_y - self.y) - 2.0 * zeta * wn * self._vy
        self._vx += ax * dt
        self._vy += ay * dt
        self.x += self._vx * dt
        self.y += self._vy * dt
