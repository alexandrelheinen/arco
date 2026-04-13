"""Rigid body models for object-centric control."""

from __future__ import annotations

from arco.control.rigid_body.base import RigidBody
from arco.control.rigid_body.circle import CircleBody
from arco.control.rigid_body.square import SquareBody

__all__ = ["CircleBody", "RigidBody", "SquareBody"]
