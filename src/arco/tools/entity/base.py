"""Geometry descriptors for physical entities.

Provides the abstract :class:`Entity` base and two concrete geometry
types: :class:`BoxGeometry` (rectangle in 2-D, cuboid in 3-D) and
:class:`SphereGeometry` (circle in 2-D, sphere in 3-D).

All classes are JSON-serialisable dataclasses.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Geometry descriptors
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Geometry(ABC):
    """Abstract base for entity geometry.

    Geometry determines how an entity's collision footprint and visual
    representation are computed.  Concrete subclasses must implement
    :meth:`to_dict` and :meth:`from_dict`.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise geometry to a JSON-safe dictionary.

        Returns:
            A plain dict whose values are JSON primitives.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Geometry:
        """Deserialise geometry from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`Geometry` instance.
        """


@dataclasses.dataclass
class BoxGeometry(Geometry):
    """Axis-aligned rectangular (2-D) or cuboidal (3-D) geometry.

    Attributes:
        half_extents: Half-widths along each dimension ``(hx,)`` for 1-D,
            ``(hx, hy)`` for 2-D, or ``(hx, hy, hz)`` for 3-D.
    """

    half_extents: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with keys ``type`` and ``half_extents``.
        """
        return {
            "type": "box",
            "half_extents": list(self.half_extents),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoxGeometry:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict with ``half_extents`` as a list of floats.

        Returns:
            A :class:`BoxGeometry` instance.

        Raises:
            KeyError: If ``half_extents`` is absent.
        """
        return cls(half_extents=tuple(float(v) for v in data["half_extents"]))


@dataclasses.dataclass
class SphereGeometry(Geometry):
    """Circular (2-D) or spherical (3-D) geometry.

    Attributes:
        radius: Radius in metres.
    """

    radius: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with keys ``type`` and ``radius``.
        """
        return {"type": "sphere", "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SphereGeometry:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict with ``radius`` as a float.

        Returns:
            A :class:`SphereGeometry` instance.

        Raises:
            KeyError: If ``radius`` is absent.
        """
        return cls(radius=float(data["radius"]))


def geometry_from_dict(data: dict[str, Any]) -> Geometry:
    """Dispatch-deserialise a geometry from its serialised dict.

    Args:
        data: Dict with a ``type`` key set to ``"box"`` or ``"sphere"``.

    Returns:
        The matching :class:`Geometry` subclass instance.

    Raises:
        ValueError: If ``type`` is unknown.
        KeyError: If ``type`` is missing.
    """
    kind = data["type"]
    if kind == "box":
        return BoxGeometry.from_dict(data)
    if kind == "sphere":
        return SphereGeometry.from_dict(data)
    raise ValueError(f"Unknown geometry type: {kind!r}")


# ---------------------------------------------------------------------------
# Abstract entity base
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Entity(ABC):
    """Abstract base for all physical entities in an ARCO scene.

    An entity has a name, a geometry, and a mutable state vector.  Concrete
    subclasses add domain-specific attributes (actuators, kinematic parents,
    etc.).

    All entities are JSON-serialisable via :meth:`to_dict` / :meth:`from_dict`.

    Attributes:
        name: Human-readable identifier, unique within a scene.
        geometry: Collision/visual geometry descriptor.
        state: Mutable state vector as a list of floats.  Interpretation
            depends on the subclass (e.g. ``[x, y, heading]`` for a Dubins
            agent, ``[x, y, z]`` for a 3-D object).
    """

    name: str
    geometry: Geometry
    state: list[float] = dataclasses.field(default_factory=list)

    @property
    def position(self) -> list[float]:
        """Return the position portion of :attr:`state`.

        By convention, position occupies the first ``dim`` elements of
        ``state`` where ``dim`` is inferred from :attr:`geometry`.

        Returns:
            Position sub-vector as a list of floats.
        """
        return self.state[: self._position_dim]

    @property
    def _position_dim(self) -> int:
        """Infer the spatial dimension from geometry and state.

        Returns:
            Number of position coordinates: length of half-extents for box
            geometry, or ``len(state) - 1`` for sphere geometry (reserving
            the last element for heading), with a minimum of 2.
        """
        if isinstance(self.geometry, BoxGeometry):
            # Box geometry: dimension equals the number of half-extent axes.
            return len(self.geometry.half_extents)
        # Sphere geometry: last state element is heading; position is the rest.
        if not self.state:
            return 2
        return max(2, len(self.state) - 1)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise entity to a JSON-safe dictionary.

        Returns:
            A plain dict suitable for :func:`json.dumps`.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Deserialise entity from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            An entity instance.
        """
