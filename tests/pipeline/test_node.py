"""Tests for PipelineNode: thread lifecycle, bus wiring, stop behavior."""

from __future__ import annotations

import queue
import threading
import time

import pytest

from arco.middleware.bus import InMemoryBus
from arco.middleware.types import MappingFrame
from arco.pipeline.node import PipelineNode

# ---------------------------------------------------------------------------
# Concrete node fixtures
# ---------------------------------------------------------------------------


class _CountingNode(PipelineNode):
    """Node that publishes a fixed number of MappingFrame instances and exits."""

    def __init__(self, count: int = 3) -> None:
        super().__init__(name="counting")
        self.published: list[MappingFrame] = []
        self._count = count

    def run(self) -> None:
        for i in range(self._count):
            if self.stop_requested:
                break
            frame = MappingFrame(timestamp=float(i))
            self.published.append(frame)
            self.publish(frame)


class _LoopingNode(PipelineNode):
    """Node that loops until stop is requested."""

    def __init__(self) -> None:
        super().__init__(name="looping")
        self.iteration_count = 0

    def run(self) -> None:
        while not self.stop_requested:
            self.iteration_count += 1
            time.sleep(0.005)


class _ErrorNode(PipelineNode):
    """Node whose run() raises an exception (must not crash the process)."""

    def __init__(self) -> None:
        super().__init__(name="error")

    def run(self) -> None:
        raise RuntimeError("intentional error for testing")


# ---------------------------------------------------------------------------
# name / properties
# ---------------------------------------------------------------------------


def test_node_name():
    node = _CountingNode()
    assert node.name == "counting"


def test_node_not_running_before_start():
    node = _CountingNode()
    assert not node.is_running


def test_stop_requested_false_before_stop():
    node = _CountingNode()
    assert not node.stop_requested


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


def test_node_starts_background_thread():
    node = _LoopingNode()
    node.start()
    time.sleep(0.02)
    assert node.is_running
    node.stop(timeout=2.0)


def test_node_is_not_running_after_stop():
    node = _LoopingNode()
    node.start()
    time.sleep(0.02)
    node.stop(timeout=2.0)
    assert not node.is_running


def test_start_twice_is_idempotent():
    node = _LoopingNode()
    node.start()
    node.start()  # Second call must be a no-op.
    time.sleep(0.02)
    # There should be exactly one live thread.
    assert node.is_running
    node.stop(timeout=2.0)


def test_stop_before_start_is_no_op():
    node = _CountingNode()
    node.stop()  # Must not raise.
    assert not node.is_running


def test_node_can_be_restarted():
    node = _LoopingNode()
    node.start()
    time.sleep(0.02)
    node.stop(timeout=2.0)
    assert not node.is_running

    node.start()
    time.sleep(0.02)
    assert node.is_running
    node.stop(timeout=2.0)


def test_finite_node_exits_on_completion():
    node = _CountingNode(count=3)
    node.start()
    # Wait for the thread to finish naturally.
    node._thread.join(timeout=2.0)
    assert not node.is_running


# ---------------------------------------------------------------------------
# Bus wiring
# ---------------------------------------------------------------------------


def test_node_publishes_to_attached_bus():
    bus = InMemoryBus()
    q = bus.subscribe(MappingFrame)

    node = _CountingNode(count=3)
    node.attach_bus(bus)
    node.start()
    node._thread.join(timeout=2.0)

    received = []
    while True:
        try:
            received.append(q.get_nowait())
        except queue.Empty:
            break

    assert len(received) == 3
    assert [f.timestamp for f in received] == [0.0, 1.0, 2.0]


def test_node_without_bus_does_not_raise():
    node = _CountingNode(count=2)
    node.start()
    node._thread.join(timeout=2.0)
    # If no bus is attached, publish is a no-op; must not raise.


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_node_exception_does_not_crash_process():
    node = _ErrorNode()
    node.start()
    node._thread.join(timeout=2.0)
    assert not node.is_running  # Thread exited after the exception.


# ---------------------------------------------------------------------------
# stop_requested propagation
# ---------------------------------------------------------------------------


def test_stop_requested_set_by_stop():
    node = _LoopingNode()
    node.start()
    time.sleep(0.02)
    node.stop(timeout=2.0)
    assert node.stop_requested


def test_looping_node_respects_stop_request():
    node = _LoopingNode()
    node.start()
    time.sleep(0.05)
    count_before = node.iteration_count
    node.stop(timeout=2.0)
    count_after = node.iteration_count
    # Node must have stopped; iteration count must not grow further.
    time.sleep(0.05)
    assert node.iteration_count == count_after
    assert count_before > 0
