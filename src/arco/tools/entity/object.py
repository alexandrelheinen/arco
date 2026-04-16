"""Passive manipulated object entity.

An :class:`Object` is a body with no actuators that can only be moved
through contact forces applied by an agent or end-effector.

Geometry may be :class:`~arco.tools.entity.base.BoxGeometry`
(rectangle/cuboid) or :class:`~arco.tools.entity.base.SphereGeometry`
(circle/sphere).  The choice determines how contacts and collisions are
resolved.

All classes are JSON-serialisable dataclasses.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .base import Entity, Geometry, geometry_from_dict


@dataclasses.dataclass
class Object(Entity):
    """Passive manipulated body with no actuators.

    An object cannot command its own motion; it must be moved by external
    contact forces.  Its state tracks position and orientation; its geometry
    determines how contacts are computed.

    State vector: ``[x, y, heading]`` for 2-D, ``[x, y, z, rx, ry, rz]``
    for 3-D (position + Euler angles in radians).

    Attributes:
        name: Human-readable identifier, unique within a scene.
        geometry: Collision/visual geometry descriptor.
        state: Mutable pose vector (position + orientation).
        mass: Mass in kilograms.  Used when computing contact impulses.
    """

    mass: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``,
            and ``mass``.
        """
        return {
            "type": "Object",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "mass": self.mass,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Object:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            An :class:`Object` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [])),
            mass=float(data.get("mass", 1.0)),
        )
