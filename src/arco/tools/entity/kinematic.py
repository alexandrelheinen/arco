"""Kinematic chain entities: Joint, Link, EndEffector, KinematicChain.

Provides a typed hierarchy for robot kinematic structures:

* :class:`Joint` (abstract) — kinematic connection between two links.
* :class:`RevoluteJoint` — single-axis rotation (e.g. elbow, wrist).
* :class:`PrismaticJoint` — single-axis translation (e.g. linear slide).
* :class:`Link` — rigid body segment connecting joints.
* :class:`EndEffector` — terminal element of a kinematic chain.
* :class:`KinematicChain` — ordered assembly of links and joints.

All classes are JSON-serialisable dataclasses.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Any

from .base import Entity, Geometry, geometry_from_dict

# ---------------------------------------------------------------------------
# Joint types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Joint(Entity):
    """Abstract kinematic joint connecting a parent link to a child link.

    Attributes:
        name: Human-readable identifier.
        geometry: Visual geometry of the joint housing.
        state: ``[position]`` — current joint position.  Radians for
            revolute joints; metres for prismatic joints.
        min_position: Lower bound on joint position.
        max_position: Upper bound on joint position.
        parent_link: Name of the parent link in the kinematic chain.
        child_link: Name of the child link in the kinematic chain.
    """

    state: list[float] = dataclasses.field(default_factory=lambda: [0.0])
    min_position: float = -float("inf")
    max_position: float = float("inf")
    parent_link: str = ""
    child_link: str = ""

    @property
    def position(self) -> float:
        """Current joint position (scalar).

        Returns:
            First element of :attr:`state`.
        """
        return self.state[0] if self.state else 0.0

    @position.setter
    def position(self, value: float) -> None:
        """Set joint position, clamped to ``[min_position, max_position]``.

        Args:
            value: Desired joint position.
        """
        clamped = max(self.min_position, min(self.max_position, value))
        if self.state:
            self.state[0] = clamped
        else:
            self.state = [clamped]

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            A plain dict suitable for :func:`json.dumps`.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Joint:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`Joint` instance.
        """


@dataclasses.dataclass
class RevoluteJoint(Joint):
    """Single-axis rotational joint.

    Joint position is in radians.

    Attributes:
        name: Human-readable identifier.
        geometry: Visual geometry.
        state: ``[angle]`` in radians.
        min_position: Minimum angle in radians.
        max_position: Maximum angle in radians.
        parent_link: Parent link name.
        child_link: Child link name.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``,
            ``min_position``, ``max_position``, ``parent_link``, and
            ``child_link``.
        """
        return {
            "type": "RevoluteJoint",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "min_position": self.min_position,
            "max_position": self.max_position,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RevoluteJoint:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`RevoluteJoint` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [0.0])),
            min_position=float(data.get("min_position", -float("inf"))),
            max_position=float(data.get("max_position", float("inf"))),
            parent_link=data.get("parent_link", ""),
            child_link=data.get("child_link", ""),
        )


@dataclasses.dataclass
class PrismaticJoint(Joint):
    """Single-axis translational joint.

    Joint position is in metres.

    Attributes:
        name: Human-readable identifier.
        geometry: Visual geometry.
        state: ``[displacement]`` in metres.
        min_position: Minimum displacement in metres.
        max_position: Maximum displacement in metres.
        parent_link: Parent link name.
        child_link: Child link name.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``,
            ``min_position``, ``max_position``, ``parent_link``, and
            ``child_link``.
        """
        return {
            "type": "PrismaticJoint",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "min_position": self.min_position,
            "max_position": self.max_position,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrismaticJoint:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`PrismaticJoint` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [0.0])),
            min_position=float(data.get("min_position", -float("inf"))),
            max_position=float(data.get("max_position", float("inf"))),
            parent_link=data.get("parent_link", ""),
            child_link=data.get("child_link", ""),
        )


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Link(Entity):
    """Rigid body segment in a kinematic chain.

    A link connects a parent joint to one or more child joints.  Its geometry
    describes the physical shape used for collision and visualisation.

    State vector: pose in world frame ``[x, y, heading]`` for 2-D or
    ``[x, y, z, rx, ry, rz]`` for 3-D.

    Attributes:
        name: Human-readable identifier.
        geometry: Collision/visual geometry.
        state: Pose in world frame (position + orientation).
        mass: Link mass in kilograms.
        parent_joint: Name of the joint that drives this link (empty for the
            base link).
    """

    mass: float = 0.0
    parent_joint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``, ``mass``,
            and ``parent_joint``.
        """
        return {
            "type": "Link",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "mass": self.mass,
            "parent_joint": self.parent_joint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Link:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`Link` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [])),
            mass=float(data.get("mass", 0.0)),
            parent_joint=data.get("parent_joint", ""),
        )


# ---------------------------------------------------------------------------
# EndEffector
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EndEffector(Entity):
    """Terminal element of a kinematic chain.

    The end-effector is the point of interaction with the environment
    (e.g. gripper, tool centre point).  It has no children.

    State vector: pose in world frame ``[x, y, heading]`` for 2-D or
    ``[x, y, z, rx, ry, rz]`` for 3-D.

    Attributes:
        name: Human-readable identifier.
        geometry: Collision/visual geometry.
        state: Pose in world frame.
        parent_joint: Name of the joint that drives this end-effector.
    """

    parent_joint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``type``, ``name``, ``geometry``, ``state``, and
            ``parent_joint``.
        """
        return {
            "type": "EndEffector",
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "state": list(self.state),
            "parent_joint": self.parent_joint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EndEffector:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            An :class:`EndEffector` instance.
        """
        return cls(
            name=data["name"],
            geometry=geometry_from_dict(data["geometry"]),
            state=list(data.get("state", [])),
            parent_joint=data.get("parent_joint", ""),
        )


# ---------------------------------------------------------------------------
# Kinematic chain
# ---------------------------------------------------------------------------


def _joint_from_dict(data: dict[str, Any]) -> Joint:
    """Dispatch-deserialise a joint from its serialised dict.

    Args:
        data: Dict with a ``type`` key set to ``"RevoluteJoint"`` or
            ``"PrismaticJoint"``.

    Returns:
        The matching :class:`Joint` subclass instance.

    Raises:
        ValueError: If ``type`` is unknown.
        KeyError: If ``type`` is missing.
    """
    kind = data["type"]
    if kind == "RevoluteJoint":
        return RevoluteJoint.from_dict(data)
    if kind == "PrismaticJoint":
        return PrismaticJoint.from_dict(data)
    raise ValueError(f"Unknown joint type: {kind!r}")


@dataclasses.dataclass
class KinematicChain:
    """Ordered assembly of links, joints, and an end-effector.

    Provides a lightweight container that groups the components of a serial
    kinematic chain.  The physical relationship between components (i.e.
    which link is attached to which joint) is encoded in each entity's
    ``parent_joint`` / ``parent_link`` / ``child_link`` attributes.

    Attributes:
        name: Human-readable chain identifier.
        links: Ordered list of rigid-body segments, base-to-tip.
        joints: Ordered list of joints, base-to-tip.
        end_effector: Terminal element.
    """

    name: str
    links: list[Link] = dataclasses.field(default_factory=list)
    joints: list[Joint] = dataclasses.field(default_factory=list)
    end_effector: EndEffector | None = None

    @property
    def dof(self) -> int:
        """Degrees of freedom — number of joints.

        Returns:
            Number of joints in the chain.
        """
        return len(self.joints)

    @property
    def joint_positions(self) -> list[float]:
        """Current joint positions.

        Returns:
            List of joint positions in the same order as :attr:`joints`.
        """
        return [j.position for j in self.joints]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dict with ``name``, ``links``, ``joints``, and
            ``end_effector`` (``null`` if absent).
        """
        return {
            "name": self.name,
            "links": [lk.to_dict() for lk in self.links],
            "joints": [j.to_dict() for j in self.joints],
            "end_effector": (
                self.end_effector.to_dict()
                if self.end_effector is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KinematicChain:
        """Deserialise from a plain dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A :class:`KinematicChain` instance.
        """
        ee_data = data.get("end_effector")
        return cls(
            name=data["name"],
            links=[Link.from_dict(d) for d in data.get("links", [])],
            joints=[_joint_from_dict(d) for d in data.get("joints", [])],
            end_effector=(
                EndEffector.from_dict(ee_data) if ee_data is not None else None
            ),
        )
