"""Tests for BusPublisher mixin."""

from __future__ import annotations

import queue

import pytest

from arco.middleware.bus import InMemoryBus
from arco.middleware.publisher import BusPublisher
from arco.middleware.types import MappingFrame, PlanFrame


class _ConcretePublisher(BusPublisher):
    """Minimal concrete publisher for testing."""


# ---------------------------------------------------------------------------
# attach_bus / publish
# ---------------------------------------------------------------------------


def test_publish_without_bus_is_no_op():
    publisher = _ConcretePublisher()
    # Must not raise even though no bus is attached.
    publisher.publish(MappingFrame(timestamp=0.0))


def test_publish_with_bus_delivers_frame():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)

    publisher = _ConcretePublisher()
    publisher.attach_bus(bus)

    frame = MappingFrame(timestamp=1.0)
    publisher.publish(frame)

    assert q.get_nowait() is frame


def test_attach_bus_replaces_previous_bus():
    bus1 = InMemoryBus()
    bus2 = InMemoryBus()
    q1 = bus1.subscribe(MappingFrame)
    q2 = bus2.subscribe(MappingFrame)

    publisher = _ConcretePublisher()
    publisher.attach_bus(bus1)
    publisher.attach_bus(bus2)

    publisher.publish(MappingFrame(timestamp=1.0))

    # Only bus2's queue should receive the frame.
    with pytest.raises(queue.Empty):
        q1.get_nowait()
    assert q2.qsize() == 1


def test_publisher_routes_different_frame_types():
    bus = InMemoryBus()
    mapping_q = bus.subscribe(MappingFrame)
    plan_q = bus.subscribe(PlanFrame)

    publisher = _ConcretePublisher()
    publisher.attach_bus(bus)

    publisher.publish(MappingFrame(timestamp=1.0))
    publisher.publish(PlanFrame(timestamp=2.0))

    assert mapping_q.qsize() == 1
    assert plan_q.qsize() == 1


def test_multiple_publishers_share_same_bus():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)

    p1 = _ConcretePublisher()
    p1.attach_bus(bus)
    p2 = _ConcretePublisher()
    p2.attach_bus(bus)

    p1.publish(MappingFrame(timestamp=1.0))
    p2.publish(MappingFrame(timestamp=2.0))

    assert q.qsize() == 2
