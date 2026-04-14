"""Tests for BusSubscriber mixin."""

from __future__ import annotations

import time

from arco.middleware.bus import InMemoryBus
from arco.middleware.subscriber import BusSubscriber
from arco.middleware.types import GuidanceFrame, MappingFrame, PlanFrame


class _ConcreteSubscriber(BusSubscriber):
    """Minimal concrete subscriber for testing."""


# ---------------------------------------------------------------------------
# subscribe / next_frame
# ---------------------------------------------------------------------------


def test_next_frame_returns_none_when_not_subscribed():
    sub = _ConcreteSubscriber()
    assert sub.next_frame(MappingFrame) is None


def test_next_frame_returns_none_on_empty_queue():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)
    assert sub.next_frame(MappingFrame) is None


def test_next_frame_returns_published_frame():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    frame = MappingFrame(timestamp=1.0)
    bus.publish(frame)

    received = sub.next_frame(MappingFrame)
    assert received is frame


def test_next_frame_fifo_order():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    f1 = MappingFrame(timestamp=1.0)
    f2 = MappingFrame(timestamp=2.0)
    bus.publish(f1)
    bus.publish(f2)

    assert sub.next_frame(MappingFrame) is f1
    assert sub.next_frame(MappingFrame) is f2


def test_next_frame_blocking_receives_frame():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    frame = MappingFrame(timestamp=3.0)
    bus.publish(frame)

    received = sub.next_frame(MappingFrame, block=True, timeout=1.0)
    assert received is frame


def test_next_frame_blocking_timeout_returns_none():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    start = time.monotonic()
    result = sub.next_frame(MappingFrame, block=True, timeout=0.05)
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed >= 0.05


def test_subscribe_multiple_types():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)
    sub.subscribe(bus, PlanFrame)

    bus.publish(MappingFrame(timestamp=1.0))
    bus.publish(PlanFrame(timestamp=2.0))

    assert sub.next_frame(MappingFrame) is not None
    assert sub.next_frame(PlanFrame) is not None


# ---------------------------------------------------------------------------
# drain_latest
# ---------------------------------------------------------------------------


def test_drain_latest_returns_none_when_not_subscribed():
    sub = _ConcreteSubscriber()
    assert sub.drain_latest(MappingFrame) is None


def test_drain_latest_returns_none_on_empty_queue():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)
    assert sub.drain_latest(MappingFrame) is None


def test_drain_latest_returns_most_recent_frame():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    f1 = MappingFrame(timestamp=1.0)
    f2 = MappingFrame(timestamp=2.0)
    f3 = MappingFrame(timestamp=3.0)
    bus.publish(f1)
    bus.publish(f2)
    bus.publish(f3)

    latest = sub.drain_latest(MappingFrame)
    assert latest is f3


def test_drain_latest_empties_queue():
    bus = InMemoryBus()
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    for i in range(5):
        bus.publish(MappingFrame(timestamp=float(i)))

    sub.drain_latest(MappingFrame)

    # Queue must be empty after draining.
    assert sub.next_frame(MappingFrame) is None


# ---------------------------------------------------------------------------
# Late subscriber behavior
# ---------------------------------------------------------------------------


def test_late_subscriber_misses_early_frames():
    bus = InMemoryBus()

    # Publish before subscriber registers.
    bus.publish(MappingFrame(timestamp=0.0))
    bus.publish(MappingFrame(timestamp=1.0))

    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    # Must not receive pre-registration frames.
    assert sub.next_frame(MappingFrame) is None


def test_late_subscriber_receives_post_registration_frames():
    bus = InMemoryBus()

    # Publish before subscriber registers.
    bus.publish(MappingFrame(timestamp=0.0))

    sub = _ConcreteSubscriber()
    sub.subscribe(bus, MappingFrame)

    # Post-registration frame must arrive.
    late_frame = MappingFrame(timestamp=1.0)
    bus.publish(late_frame)
    assert sub.next_frame(MappingFrame) is late_frame


# ---------------------------------------------------------------------------
# Frame dropping on slow consumer
# ---------------------------------------------------------------------------


def test_slow_consumer_drops_frames_without_blocking_producer():
    """A full subscriber queue must not block the producer."""
    bus = InMemoryBus(maxsize=2)
    sub = _ConcreteSubscriber()
    sub.subscribe(bus, GuidanceFrame)

    # Flood the bus with more frames than the queue can hold.
    for i in range(10):
        bus.publish(GuidanceFrame(timestamp=float(i)))

    # Consumer can only retrieve up to maxsize frames.
    received = []
    while True:
        f = sub.next_frame(GuidanceFrame)
        if f is None:
            break
        received.append(f)

    assert len(received) == 2
    # The frames received are the first two (FIFO); later ones were dropped.
    assert received[0].timestamp == 0.0
    assert received[1].timestamp == 1.0
