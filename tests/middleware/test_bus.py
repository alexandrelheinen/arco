"""Tests for InMemoryBus: publish, subscribe, frame dropping, late subscriber."""

from __future__ import annotations

import queue
import threading
import time

import pytest

from arco.middleware.bus import InMemoryBus
from arco.middleware.types import GuidanceFrame, MappingFrame, PlanFrame


# ---------------------------------------------------------------------------
# Basic publish / subscribe
# ---------------------------------------------------------------------------


def test_subscribe_returns_queue():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)
    assert isinstance(q, queue.Queue)


def test_published_frame_arrives_in_queue():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)
    frame = MappingFrame(timestamp=1.0)
    bus.publish(frame)
    received = q.get_nowait()
    assert received is frame


def test_multiple_subscribers_each_receive_frame():
    bus = InMemoryBus()
    q1 = bus.subscribe(MappingFrame)
    q2 = bus.subscribe(MappingFrame)
    frame = MappingFrame(timestamp=2.0)
    bus.publish(frame)
    assert q1.get_nowait() is frame
    assert q2.get_nowait() is frame


def test_publish_routes_by_type():
    bus = InMemoryBus()
    mapping_q = bus.subscribe(MappingFrame)
    plan_q = bus.subscribe(PlanFrame)

    mapping_frame = MappingFrame(timestamp=1.0)
    plan_frame = PlanFrame(timestamp=2.0)
    bus.publish(mapping_frame)
    bus.publish(plan_frame)

    assert mapping_q.get_nowait() is mapping_frame
    assert plan_q.get_nowait() is plan_frame

    # Cross-type queues must remain empty.
    with pytest.raises(queue.Empty):
        mapping_q.get_nowait()
    with pytest.raises(queue.Empty):
        plan_q.get_nowait()


def test_publish_without_subscribers_is_no_op():
    bus = InMemoryBus()
    # No subscriber registered; must not raise.
    bus.publish(MappingFrame(timestamp=0.0))


# ---------------------------------------------------------------------------
# subscriber_count
# ---------------------------------------------------------------------------


def test_subscriber_count_zero_before_subscribe():
    bus = InMemoryBus()
    assert bus.subscriber_count(MappingFrame) == 0


def test_subscriber_count_increments():
    bus = InMemoryBus()
    bus.subscribe(MappingFrame)
    assert bus.subscriber_count(MappingFrame) == 1
    bus.subscribe(MappingFrame)
    assert bus.subscriber_count(MappingFrame) == 2


# ---------------------------------------------------------------------------
# Frame dropping under a slow consumer
# ---------------------------------------------------------------------------


def test_frame_dropped_when_queue_full():
    """Producer must never block; extra frames are silently dropped."""
    bus = InMemoryBus(maxsize=2)
    q = bus.subscribe(MappingFrame)

    # Publish more frames than the queue can hold.
    for i in range(5):
        bus.publish(MappingFrame(timestamp=float(i)))

    # Queue holds at most 2 frames; the rest were silently dropped.
    count = 0
    while True:
        try:
            q.get_nowait()
            count += 1
        except queue.Empty:
            break
    assert count == 2


def test_producer_not_blocked_by_full_queue():
    """Publishing to a full queue must return in negligible time."""
    bus = InMemoryBus(maxsize=1)
    bus.subscribe(MappingFrame)  # Don't consume the queue.

    start = time.monotonic()
    for _ in range(100):
        bus.publish(MappingFrame(timestamp=0.0))
    elapsed = time.monotonic() - start

    # 100 publish calls must complete in well under 1 second.
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Late subscriber
# ---------------------------------------------------------------------------


def test_late_subscriber_receives_frames_published_after_subscribe():
    bus = InMemoryBus()

    # Publish before subscriber exists.
    bus.publish(MappingFrame(timestamp=0.0))

    # Late subscriber registers now.
    q = bus.subscribe(MappingFrame)

    # The pre-registration frame must NOT be in the queue.
    with pytest.raises(queue.Empty):
        q.get_nowait()

    # Post-registration frames ARE received.
    frame = MappingFrame(timestamp=1.0)
    bus.publish(frame)
    assert q.get_nowait() is frame


def test_late_subscriber_does_not_affect_existing_subscribers():
    bus = InMemoryBus()
    q_early = bus.subscribe(MappingFrame)

    bus.publish(MappingFrame(timestamp=1.0))

    # Late subscriber joins.
    q_late = bus.subscribe(MappingFrame)

    frame2 = MappingFrame(timestamp=2.0)
    bus.publish(frame2)

    # Early subscriber received both frames.
    assert q_early.qsize() == 2
    # Late subscriber received only the second frame.
    assert q_late.qsize() == 1
    assert q_late.get_nowait() is frame2


# ---------------------------------------------------------------------------
# unsubscribe
# ---------------------------------------------------------------------------


def test_unsubscribe_removes_queue():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)
    assert bus.subscriber_count(MappingFrame) == 1

    bus.unsubscribe(MappingFrame, q)
    assert bus.subscriber_count(MappingFrame) == 0

    # Published frames must no longer reach the removed queue.
    bus.publish(MappingFrame(timestamp=0.0))
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_unsubscribe_unknown_queue_is_no_op():
    bus = InMemoryBus()
    stray: queue.Queue = queue.Queue()
    # Must not raise.
    bus.unsubscribe(MappingFrame, stray)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_publish_and_subscribe():
    """Multiple threads publishing and subscribing must not corrupt state."""
    bus = InMemoryBus(maxsize=0)  # Unbounded to avoid drops in this test.
    frame_count = 50
    subscriber_count = 5

    queues = [bus.subscribe(MappingFrame) for _ in range(subscriber_count)]
    received = [0] * subscriber_count

    errors = []

    def producer():
        for i in range(frame_count):
            bus.publish(MappingFrame(timestamp=float(i)))

    def consumer(idx: int):
        for _ in range(frame_count):
            try:
                queues[idx].get(timeout=2.0)
                received[idx] += 1
            except queue.Empty:
                errors.append(f"consumer {idx} timed out")

    threads = [threading.Thread(target=producer)]
    threads += [
        threading.Thread(target=consumer, args=(i,))
        for i in range(subscriber_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, errors
    for i in range(subscriber_count):
        assert received[i] == frame_count
