"""Tests for PipelineRunner: node management, bus wiring, lifecycle."""

from __future__ import annotations

import queue
import time
from pathlib import Path

import pytest

from arco.middleware.bus import InMemoryBus
from arco.middleware.subscriber import BusSubscriber
from arco.middleware.types import GuidanceFrame, MappingFrame, PlanFrame
from arco.pipeline.node import PipelineNode
from arco.pipeline.runner import PipelineRunner

# ---------------------------------------------------------------------------
# Helper nodes and subscribers
# ---------------------------------------------------------------------------


class _MappingNode(PipelineNode):
    """Publishes one MappingFrame and exits."""

    def run(self) -> None:
        self.publish(MappingFrame(timestamp=time.monotonic()))


class _PlanningNode(PipelineNode):
    """Publishes one PlanFrame and exits."""

    def run(self) -> None:
        self.publish(PlanFrame(timestamp=time.monotonic(), planner="RRT*"))


class _GuidanceNode(PipelineNode):
    """Publishes one GuidanceFrame and exits."""

    def run(self) -> None:
        self.publish(
            GuidanceFrame(
                timestamp=time.monotonic(),
                trajectory=[[0.0, 0.0], [1.0, 1.0]],
                durations=[0.5],
            )
        )


class _LoopingMappingNode(PipelineNode):
    """Publishes MappingFrames in a loop until stopped."""

    def run(self) -> None:
        while not self.stop_requested:
            self.publish(MappingFrame(timestamp=time.monotonic()))
            time.sleep(0.01)


class _ConcreteSubscriber(BusSubscriber):
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Write a minimal pipeline YAML config and return its path."""
    cfg = tmp_path / "test_pipeline.yml"
    cfg.write_text(
        "scenario: test\nbounds: [0.0, 0.0, 10.0, 10.0]\nclearance: 0.5\n",
        encoding="utf-8",
    )
    return cfg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_runner_loads_config(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    assert runner.config["scenario"] == "test"
    assert runner.config["clearance"] == 0.5


def test_runner_raises_on_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PipelineRunner(tmp_path / "nonexistent.yml")


def test_runner_has_bus(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    assert isinstance(runner.bus, InMemoryBus)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------


def test_register_node_wires_bus(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    q = runner.bus.subscribe(MappingFrame)

    node = _MappingNode("mapping")
    runner.register_node(node)
    runner.start()
    node._thread.join(timeout=2.0)

    received = q.get(timeout=1.0)
    assert isinstance(received, MappingFrame)


def test_register_multiple_nodes(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    mapping_q = runner.bus.subscribe(MappingFrame)
    plan_q = runner.bus.subscribe(PlanFrame)
    guidance_q = runner.bus.subscribe(GuidanceFrame)

    runner.register_node(_MappingNode("mapping"))
    runner.register_node(_PlanningNode("planning"))
    runner.register_node(_GuidanceNode("guidance"))
    runner.start()

    # Wait for all threads to finish.
    for node in runner._nodes:
        node._thread.join(timeout=2.0)

    assert isinstance(mapping_q.get(timeout=1.0), MappingFrame)
    assert isinstance(plan_q.get(timeout=1.0), PlanFrame)
    assert isinstance(guidance_q.get(timeout=1.0), GuidanceFrame)


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


def test_start_starts_all_nodes(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    n1 = _LoopingMappingNode("m1")
    n2 = _LoopingMappingNode("m2")
    runner.register_node(n1)
    runner.register_node(n2)
    runner.start()
    time.sleep(0.02)
    assert n1.is_running
    assert n2.is_running
    runner.stop(timeout=2.0)


def test_stop_stops_all_nodes(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    n1 = _LoopingMappingNode("m1")
    n2 = _LoopingMappingNode("m2")
    runner.register_node(n1)
    runner.register_node(n2)
    runner.start()
    time.sleep(0.02)
    runner.stop(timeout=2.0)
    assert not n1.is_running
    assert not n2.is_running


def test_start_already_running_nodes_is_idempotent(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    node = _LoopingMappingNode("m")
    runner.register_node(node)
    runner.start()
    time.sleep(0.02)
    runner.start()  # Second start must be a no-op.
    assert node.is_running
    runner.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# Subscriber attachment (including late subscriber)
# ---------------------------------------------------------------------------


def test_attach_subscriber_before_start(config_path: Path) -> None:
    runner = PipelineRunner(config_path)
    sub = _ConcreteSubscriber()
    runner.attach_subscriber(sub, GuidanceFrame)

    runner.register_node(_GuidanceNode("guidance"))
    runner.start()
    for node in runner._nodes:
        node._thread.join(timeout=2.0)

    frame = sub.next_frame(GuidanceFrame, block=True, timeout=1.0)
    assert isinstance(frame, GuidanceFrame)


def test_attach_subscriber_after_start_late_subscriber(
    config_path: Path,
) -> None:
    """Late subscriber must receive frames published after attachment."""
    runner = PipelineRunner(config_path)
    looping = _LoopingMappingNode("mapping")
    runner.register_node(looping)
    runner.start()
    time.sleep(0.02)

    # Subscribe *after* the pipeline has started.
    sub = _ConcreteSubscriber()
    runner.attach_subscriber(sub, MappingFrame)

    # Give the pipeline time to publish more frames.
    time.sleep(0.05)
    runner.stop(timeout=2.0)

    frame = sub.next_frame(MappingFrame)
    assert frame is not None


# ---------------------------------------------------------------------------
# Full chain: mapping → planning → guidance without frontend
# ---------------------------------------------------------------------------


def test_full_chain_without_frontend(config_path: Path) -> None:
    """PipelineRunner runs all three stages; no frontend is needed."""
    runner = PipelineRunner(config_path)
    runner.register_node(_MappingNode("mapping"))
    runner.register_node(_PlanningNode("planning"))
    runner.register_node(_GuidanceNode("guidance"))
    runner.start()

    for node in runner._nodes:
        node._thread.join(timeout=2.0)

    # All nodes must have exited cleanly.
    for node in runner._nodes:
        assert not node.is_running


# ---------------------------------------------------------------------------
# Frame dropping does not affect pipeline
# ---------------------------------------------------------------------------


def test_frame_dropping_does_not_block_pipeline(config_path: Path) -> None:
    """A full subscriber queue must not stall the pipeline."""
    runner = PipelineRunner(config_path, bus_maxsize=1)
    looping = _LoopingMappingNode("mapping")
    runner.register_node(looping)

    # Subscribe but never consume (simulate a stalled frontend).
    _stall_q = runner.bus.subscribe(MappingFrame)

    start = time.monotonic()
    runner.start()
    time.sleep(0.1)  # Let the pipeline run for 100 ms.
    runner.stop(timeout=2.0)
    elapsed = time.monotonic() - start

    # Pipeline ran for ~100 ms; total wall-clock must be well under 1 s.
    assert elapsed < 2.0
