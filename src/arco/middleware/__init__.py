"""Shared in-memory middleware for the ARCO async pipeline.

This package provides the typed message-bus infrastructure that connects
the mapping, planning, and guidance pipeline stages to one or more
frontend renderers without coupling them.  ``arcosim --image`` (matplotlib
static image generation) and ``arcosim`` (pygame real-time simulation) both
consume the same bus and are unaware of each other.

Public API:
    - :class:`~arco.middleware.bus.Bus` — abstract bus interface.
    - :class:`~arco.middleware.bus.InMemoryBus` — thread-safe
      implementation backed by bounded :class:`queue.Queue` instances.
    - :class:`~arco.middleware.publisher.BusPublisher` — mixin for
      pipeline nodes that produce frames.
    - :class:`~arco.middleware.subscriber.BusSubscriber` — mixin for
      consumers (frontends) that receive frames.
    - :mod:`arco.middleware.types` — typed arc dataclasses
      (:class:`~arco.middleware.types.MappingFrame`,
      :class:`~arco.middleware.types.PlanFrame`,
      :class:`~arco.middleware.types.GuidanceFrame`).
"""

from .bus import Bus, InMemoryBus
from .publisher import BusPublisher
from .subscriber import BusSubscriber
from .types import GuidanceFrame, MappingFrame, PlanFrame

__all__ = [
    "Bus",
    "BusPublisher",
    "BusSubscriber",
    "GuidanceFrame",
    "InMemoryBus",
    "MappingFrame",
    "PlanFrame",
]
