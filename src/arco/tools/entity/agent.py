"""Mobile actuated agent entities.

Provides the abstract :class:`Agent` base and two concrete agent types:

* :class:`DubinsAgent` — non-holonomic vehicle (heading, speed, turn-rate).
* :class:`CartesianAgent` — N independent Cartesian axes with velocity
  control and a first-order low-pass filter modelling actuator dynamics.

All classes are JSON-serialisable dataclasses.
"""

from __future__ import annotations

import dataclasses
import math
from abc import abstractmethod
from typing import Any

from .base import Entity, Geometry, geometry_from_dict

# ---------------------------------------------------------------------------
# Abstract agent base
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Agent(Entity):
    """Abstract base for all mobile actuated agents.

    An agent extends :class:`~arco.tools.entity.base.Entity` with a
    :meth:`step` method that advances the agent's state by one time step.

    Subclasses must implement :meth:`step`, :meth:`to_dict`, and
    :meth:`from_dict`.

    Attributes:
        name: Human-readable identifier, unique within a scene.
        geometry: Collision/visual geometry descriptor.
        state: Mutable state vector.
    """

    @abstractmethod
    def step(self, control: list[float], dt: float) -> None:
        """Advance the agent state by one time step.

        Args:
            control: Control input vector.  Interpretation depends on the
                subclass (e.g. ``[turn_rate, acceleration]`` for Dubins).
            dt: Time step in seconds.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            A plain dict suitable for :func:`json.dumps`.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            An :class:`Agent` instance.
        """


# ---------------------------------------------------------------------------
# Dubins agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DubinsAgent(Agent):
    """Non-holonomic Dubins vehicle agent.

    State vector: ``[x, y, heading]`` (metres, metres, radians).
    Control vector: ``[turn_rate, acceleration]`` (rad/s, m/s²).

    Attributes:
        name: Human-readable identifier.
        geometry: Collision/visual geometry.
        state: ``[x, y, heading]`` — position and orientation.
        max_speed: Maximum forward speed in m/s.
        max_turn_rate: Maximum absolute turn-rate in rad/s.
        speed: Current forward speed in m/s.
    """

    max_speed: float = 5.0
    max_turn_rate: float = math.pi
    speed: float = 0.0

    # ------------------------------------------------------------------
    # Convenience accessors for named state elements
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """X-coordinate in metres."""
        return self.state[0]

    @x.setter
    def x(self, value: float) -> None:
        self.state[0] = value

    @property
    def y(self) -> float:
        """Y-coordinate in metres."""
        return self.state[1]

    @y.setter
    def y(self, value: float) -> None:
        self.state[1] = value

    @property
    def heading(self) -> float:
        """Heading angle in radians."""
        return self.state[2]

    @heading.setter
    def heading(self, value: float) -> None:
        self.state[2] = value

    def step(self, control: list[float], dt: float) -> None:
        """Advance Dubins kinematics by *dt* seconds.

        Applies a simple Euler integration of the unicycle model:

        .. code-block::

            heading += clamp(turn_rate, ±max_turn_rate) * dt
            speed   += acceleration * dt, clamped to [0, max_speed]
            x       += speed * cos(heading) * dt
            y       += speed * sin(heading) * dt

        Args:
            control: ``[turn_rate, acceleration]`` — turn-rate (rad/s) and
                longitudinal acceleration (m/s²).
            dt: Time step in seconds.

        Raises:
            ValueError: If ``state`` does not have at least 3 elements.
        """
        if len(self.state) < 3:
            raise ValueError(
                f"DubinsAgent '{self.name}' requires state [x, y, heading];"
                f" got {self.state}"
            )
        if len(control) < 2:
            raise ValueError(
                f"DubinsAgent control must be [turn_rate, acceleration];"
                f" got {control}"
            )
        turn_rate, accel = control[0], control[1]
        turn_rate = max(
            -self.max_turn_rate, min(self.max_turn_rate, turn_rate)
        )
        self.heading += turn_rate * dt
        self.speed = max(0.0, min(self.max_speed, self.speed + accel * dt))
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``,
            ``max_speed``, ``max_turn_rate``, and ``speed``.
        """
        return {
            "type": "DubinsAgent",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "max_speed": self.max_speed,
            "max_turn_rate": self.max_turn_rate,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DubinsAgent:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`DubinsAgent` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [])),
            max_speed=float(data.get("max_speed", 5.0)),
            max_turn_rate=float(data.get("max_turn_rate", math.pi)),
            speed=float(data.get("speed", 0.0)),
        )


# ---------------------------------------------------------------------------
# Cartesian agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CartesianAgent(Agent):
    """N-DoF Cartesian agent with velocity control and actuator dynamics.

    Each axis is driven by a first-order low-pass filter that simulates
    actuator lag.  The state vector stores joint positions; the velocity
    vector is maintained internally.

    State vector: ``[q_0, q_1, …, q_{n-1}]`` — joint positions (metres).
    Control vector: ``[v_cmd_0, v_cmd_1, …, v_cmd_{n-1}]`` — commanded
    velocities (m/s).

    Attributes:
        name: Human-readable identifier.
        geometry: Collision/visual geometry.
        state: Joint positions in metres.
        max_speed: Maximum absolute velocity per axis in m/s.
        bandwidth: First-order filter bandwidth in rad/s.  Higher values
            mean a faster (more responsive) actuator.
        velocities: Current joint velocities in m/s (internal state).
    """

    max_speed: float = 1.0
    bandwidth: float = 10.0
    velocities: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialise velocity vector to match state dimensions."""
        if not self.velocities and self.state:
            self.velocities = [0.0] * len(self.state)

    def step(self, control: list[float], dt: float) -> None:
        """Advance all Cartesian axes by *dt* seconds.

        Each axis applies a first-order lag:

        .. code-block::

            v_i += bandwidth * (v_cmd_i - v_i) * dt   (actuator dynamics)
            v_i  = clamp(v_i, ±max_speed)
            q_i += v_i * dt

        Args:
            control: Commanded velocity per axis ``[v_cmd_0, …, v_cmd_{n-1}]``.
            dt: Time step in seconds.

        Raises:
            ValueError: If the dimensions of ``control`` and ``state`` differ.
        """
        dof = len(self.state)
        if len(control) != dof:
            raise ValueError(
                f"CartesianAgent '{self.name}': control has {len(control)}"
                f" elements but state has {dof}."
            )
        if len(self.velocities) != dof:
            self.velocities = [0.0] * dof
        for i in range(dof):
            self.velocities[i] += (
                self.bandwidth * (control[i] - self.velocities[i]) * dt
            )
            self.velocities[i] = max(
                -self.max_speed, min(self.max_speed, self.velocities[i])
            )
            self.state[i] += self.velocities[i] * dt

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``,
            ``max_speed``, ``bandwidth``, and ``velocities``.
        """
        return {
            "type": "CartesianAgent",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "max_speed": self.max_speed,
            "bandwidth": self.bandwidth,
            "velocities": list(self.velocities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CartesianAgent:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`CartesianAgent` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [])),
            max_speed=float(data.get("max_speed", 1.0)),
            bandwidth=float(data.get("bandwidth", 10.0)),
            velocities=list(data.get("velocities", [])),
        )
