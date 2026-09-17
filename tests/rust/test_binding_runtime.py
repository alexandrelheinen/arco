"""The middleware and pipeline surface compiled into ``arco._arco``.

``arco.pipeline`` still ships its own Python node and runner, so the
compiled :class:`PipelineNode` and :class:`PipelineRunner` are reached
only through ``arco._arco``. The bus, the publisher and the subscriber
are re-exported under ``arco.middleware``, and the tests below use the
compiled names directly so that a future re-export does not change what
is under test.

The thread lifecycle tests poll for a state change rather than sleeping
for a fixed span, so a loaded machine slows the test down instead of
failing it. Every wait carries a bound, per ``FR-SAFE-02``.
"""

from __future__ import annotations

import pathlib
import queue
import time

import pytest

from arco._arco import (
    Bus,
    BusPublisher,
    BusSubscriber,
    InMemoryBus,
    PipelineNode,
    PipelineRunner,
)

# Longest a lifecycle assertion waits for a background thread, seconds.
MAX_WAIT = 5.0
# Gap between two polls of a background thread's state, seconds.
POLL_INTERVAL = 0.005


class MappingFrame:
    """A frame type, standing in for a dataclass a real node publishes."""

    def __init__(self, value: int) -> None:
        self.value = value


class GuidanceFrame:
    """A second frame type, so routing by class has something to miss."""


def wait_until(predicate):
    """Polls *predicate* until it holds, or gives up after MAX_WAIT.

    Args:
        predicate: A callable returning True once the wait is over.

    Returns:
        Whether the predicate held before the budget ran out.
    """
    deadline = time.monotonic() + MAX_WAIT
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_INTERVAL)
    return predicate()


class Counter(PipelineNode):
    """A node that counts iterations until it is asked to stop."""

    def __init__(self, name):
        super().__init__(name)
        self.ticks = 0

    def run(self):
        while not self.stop_requested:
            self.ticks += 1
            time.sleep(POLL_INTERVAL)


class Once(PipelineNode):
    """A node whose work finishes on its own."""

    def __init__(self, name):
        super().__init__(name)
        self.ran = False

    def run(self):
        self.ran = True


class Exploding(PipelineNode):
    """A node whose work raises, which the Python node used to hide."""

    def run(self):
        raise RuntimeError("node exploded")


# ---------------------------------------------------------------------
# The abstract base
# ---------------------------------------------------------------------


def test_the_base_bus_refuses_to_publish():
    with pytest.raises(
        NotImplementedError, match=r"Bus\.publish is abstract\."
    ):
        Bus().publish(MappingFrame(1))


def test_the_base_bus_refuses_to_register_a_subscriber():
    with pytest.raises(
        NotImplementedError, match=r"Bus\.subscribe is abstract\."
    ):
        Bus().subscribe(MappingFrame)


def test_the_base_bus_refuses_to_count_subscribers():
    with pytest.raises(
        NotImplementedError,
        match=r"Bus\.subscriber_count is abstract\.",
    ):
        Bus().subscriber_count(MappingFrame)


def test_the_base_pipeline_node_refuses_to_run():
    with pytest.raises(
        NotImplementedError, match=r"PipelineNode\.run is abstract\."
    ):
        PipelineNode("bare").run()


# ---------------------------------------------------------------------
# InMemoryBus
# ---------------------------------------------------------------------


def test_a_bus_reports_no_subscribers_for_an_unknown_frame_type():
    bus = InMemoryBus()
    assert bus.subscriber_count(MappingFrame) == 0


def test_subscribing_hands_back_a_queue_and_raises_the_count():
    bus = InMemoryBus(maxsize=4)
    channel = bus.subscribe(MappingFrame)
    assert isinstance(channel, queue.Queue)
    assert channel.maxsize == 4
    assert bus.subscriber_count(MappingFrame) == 1
    assert bus.subscriber_count(GuidanceFrame) == 0


def test_a_published_frame_reaches_every_subscriber_of_its_type():
    bus = InMemoryBus()
    first = bus.subscribe(MappingFrame)
    second = bus.subscribe(MappingFrame)
    other = bus.subscribe(GuidanceFrame)

    bus.publish(MappingFrame(7))

    assert bus.last_publish == (2, 0)
    assert first.get_nowait().value == 7
    assert second.get_nowait().value == 7
    assert other.empty()


def test_a_full_queue_drops_the_frame_and_the_bus_says_so():
    bus = InMemoryBus(maxsize=1)
    bus.subscribe(MappingFrame)

    bus.publish(MappingFrame(1))
    assert bus.last_publish == (1, 0)

    bus.publish(MappingFrame(2))
    assert bus.last_publish == (0, 1)


def test_publishing_with_no_subscriber_delivers_and_drops_nothing():
    bus = InMemoryBus()
    bus.publish(MappingFrame(1))
    assert bus.last_publish == (0, 0)


def test_unsubscribing_removes_only_the_queue_it_was_given():
    bus = InMemoryBus()
    kept = bus.subscribe(MappingFrame)
    dropped = bus.subscribe(MappingFrame)

    bus.unsubscribe(MappingFrame, dropped)

    assert bus.subscriber_count(MappingFrame) == 1
    bus.publish(MappingFrame(3))
    assert kept.get_nowait().value == 3
    assert dropped.empty()


def test_unsubscribing_a_queue_the_bus_never_held_changes_nothing():
    bus = InMemoryBus()
    registered = bus.subscribe(MappingFrame)

    bus.unsubscribe(MappingFrame, queue.Queue())
    bus.unsubscribe(GuidanceFrame, registered)

    assert bus.subscriber_count(MappingFrame) == 1


# ---------------------------------------------------------------------
# The two mixins
# ---------------------------------------------------------------------


def test_a_publisher_with_no_bus_attached_drops_the_frame_silently():
    publisher = BusPublisher()
    assert publisher.publish(MappingFrame(1)) is None


def test_an_attached_publisher_routes_through_the_bus():
    bus = InMemoryBus()
    channel = bus.subscribe(MappingFrame)
    publisher = BusPublisher()
    publisher.attach_bus(bus)

    publisher.publish(MappingFrame(11))

    assert channel.get_nowait().value == 11


def test_attaching_a_second_bus_replaces_the_first():
    first = InMemoryBus()
    second = InMemoryBus()
    abandoned = first.subscribe(MappingFrame)
    current = second.subscribe(MappingFrame)
    publisher = BusPublisher()

    publisher.attach_bus(first)
    publisher.attach_bus(second)
    publisher.publish(MappingFrame(2))

    assert abandoned.empty()
    assert current.get_nowait().value == 2


def test_a_subscriber_reports_nothing_for_a_type_it_never_took():
    subscriber = BusSubscriber()
    assert subscriber.next_frame(MappingFrame) is None
    assert subscriber.drain_latest(MappingFrame) is None


def test_polling_an_empty_queue_answers_none_instead_of_raising():
    bus = InMemoryBus()
    subscriber = BusSubscriber()
    subscriber.subscribe(bus, MappingFrame)

    assert subscriber.next_frame(MappingFrame) is None


def test_a_subscriber_takes_the_frames_in_the_order_they_arrived():
    bus = InMemoryBus()
    subscriber = BusSubscriber()
    subscriber.subscribe(bus, MappingFrame)

    bus.publish(MappingFrame(1))
    bus.publish(MappingFrame(2))

    assert subscriber.next_frame(MappingFrame).value == 1
    assert subscriber.next_frame(MappingFrame).value == 2
    assert subscriber.next_frame(MappingFrame) is None


def test_draining_keeps_the_newest_frame_and_empties_the_backlog():
    bus = InMemoryBus()
    subscriber = BusSubscriber()
    subscriber.subscribe(bus, MappingFrame)

    for value in range(5):
        bus.publish(MappingFrame(value))

    assert subscriber.drain_latest(MappingFrame).value == 4
    assert subscriber.next_frame(MappingFrame) is None


def test_a_blocking_poll_returns_none_once_its_timeout_expires():
    bus = InMemoryBus()
    subscriber = BusSubscriber()
    subscriber.subscribe(bus, MappingFrame)

    assert (
        subscriber.next_frame(MappingFrame, block=True, timeout=0.01) is None
    )


# ---------------------------------------------------------------------
# PipelineNode
# ---------------------------------------------------------------------


def test_a_fresh_node_carries_its_name_and_has_not_started():
    node = Counter("counter")
    assert node.name == "counter"
    assert node.is_running is False
    assert node.stop_requested is False
    assert node.outcome == "running"


def test_a_started_node_does_work_until_it_is_asked_to_stop():
    node = Counter("counter")
    node.start()
    assert wait_until(lambda: node.ticks > 0)
    assert node.is_running is True

    node.stop(timeout=MAX_WAIT)

    assert node.is_running is False
    assert node.stop_requested is True
    assert node.outcome == "completed"


def test_starting_a_running_node_a_second_time_does_nothing():
    node = Counter("counter")
    node.start()
    assert wait_until(lambda: node.is_running)

    node.start()

    assert node.is_running is True
    node.stop(timeout=MAX_WAIT)


def test_a_stopped_node_can_be_started_again():
    node = Counter("counter")
    node.start()
    assert wait_until(lambda: node.ticks > 0)
    node.stop(timeout=MAX_WAIT)
    first_pass = node.ticks

    node.start()
    assert wait_until(lambda: node.ticks > first_pass)
    node.stop(timeout=MAX_WAIT)

    assert node.outcome == "completed"


def test_stopping_a_node_that_never_started_is_accepted():
    node = Counter("counter")
    assert node.stop() is None
    assert node.is_running is False


def test_a_node_whose_work_returns_reports_that_it_completed():
    node = Once("once")
    node.start()
    assert wait_until(lambda: node.outcome == "completed")
    assert node.ran is True
    node.stop(timeout=MAX_WAIT)


def test_a_node_whose_work_raises_reports_the_failure_it_raised():
    node = Exploding("boom")
    node.start()
    assert wait_until(lambda: node.outcome.startswith("failed"))

    assert "node exploded" in node.outcome
    assert "boom" in node.outcome
    node.stop(timeout=MAX_WAIT)


def test_a_node_publishes_through_the_bus_it_was_attached_to():
    bus = InMemoryBus()
    channel = bus.subscribe(MappingFrame)
    node = Counter("counter")
    node.attach_bus(bus)

    node.publish(MappingFrame(42))

    assert channel.get_nowait().value == 42


# ---------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------


def write_config(directory, text):
    """Writes *text* into a pipeline configuration file.

    Args:
        directory: Where the file is written.
        text: The YAML body of the file.

    Returns:
        The path the configuration was written to.
    """
    path = pathlib.Path(directory, "pipeline.yml")
    path.write_text(text, encoding="utf-8")
    return path


def test_a_runner_refuses_a_configuration_path_that_does_not_exist(
    tmp_path,
):
    missing = tmp_path / "absent.yml"
    with pytest.raises(FileNotFoundError, match="Pipeline config not found"):
        PipelineRunner(str(missing))


def test_a_runner_reads_the_mapping_its_configuration_declares(tmp_path):
    path = write_config(tmp_path, "pipeline:\n  rate: 10\n")
    runner = PipelineRunner(path)
    assert runner.config == {"pipeline": {"rate": 10}}


def test_a_configuration_that_is_not_a_mapping_reads_as_empty(tmp_path):
    assert PipelineRunner(write_config(tmp_path, "")).config == {}
    assert PipelineRunner(write_config(tmp_path, "42\n")).config == {}


def test_the_runner_sizes_the_bus_it_owns(tmp_path):
    path = write_config(tmp_path, "pipeline: {}\n")
    assert PipelineRunner(path, bus_maxsize=3).bus.maxsize == 3


def test_registering_a_node_wires_it_to_the_shared_bus(tmp_path):
    runner = PipelineRunner(write_config(tmp_path, "pipeline: {}\n"))
    channel = runner.bus.subscribe(MappingFrame)
    node = Counter("mapping")

    runner.register_node(node)

    assert [registered.name for registered in runner.nodes] == ["mapping"]
    node.publish(MappingFrame(5))
    assert channel.get_nowait().value == 5


def test_attaching_a_subscriber_registers_it_on_the_runner_bus(tmp_path):
    runner = PipelineRunner(write_config(tmp_path, "pipeline: {}\n"))
    subscriber = BusSubscriber()

    runner.attach_subscriber(subscriber, MappingFrame)

    assert runner.bus.subscriber_count(MappingFrame) == 1
    runner.bus.publish(MappingFrame(8))
    assert subscriber.next_frame(MappingFrame).value == 8


def test_the_runner_starts_and_stops_every_node_it_registered(tmp_path):
    runner = PipelineRunner(write_config(tmp_path, "pipeline: {}\n"))
    first = Counter("first")
    second = Counter("second")
    runner.register_node(first)
    runner.register_node(second)

    runner.start()
    assert wait_until(lambda: first.ticks > 0 and second.ticks > 0)
    assert first.is_running and second.is_running

    runner.stop(timeout=MAX_WAIT)

    assert not first.is_running
    assert not second.is_running


def test_starting_a_runner_twice_leaves_the_running_nodes_alone(
    tmp_path,
):
    runner = PipelineRunner(write_config(tmp_path, "pipeline: {}\n"))
    node = Counter("only")
    runner.register_node(node)

    runner.start()
    assert wait_until(lambda: node.is_running)
    runner.start()

    assert node.is_running
    runner.stop(timeout=MAX_WAIT)
