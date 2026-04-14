"""BusPublisher: mixin that equips a pipeline node with bus publishing.

Import and inherit from :class:`BusPublisher` to give any class the
ability to publish typed frames to the shared in-memory bus.

Example::

    class MappingNode(PipelineNode, BusPublisher):
        def run(self) -> None:
            frame = MappingFrame(timestamp=time.monotonic(), ...)
            self.publish(frame)
"""

from __future__ import annotations

from typing import Optional

from .bus import Bus


class BusPublisher:
    """Mixin that adds bus-publish capability to a pipeline node.

    Attach a :class:`~arco.middleware.bus.Bus` instance via
    :meth:`attach_bus` before calling :meth:`publish`.  If no bus is
    attached, :meth:`publish` is a silent no-op so that nodes can be
    unit-tested without a live bus.
    """

    def __init__(self) -> None:
        """Initialize the publisher with no bus attached."""
        self._bus: Optional[Bus] = None

    def attach_bus(self, bus: Bus) -> None:
        """Attach a bus to this publisher.

        Args:
            bus: The shared :class:`~arco.middleware.bus.Bus` instance
                that published frames will be routed through.
        """
        self._bus = bus

    def publish(self, frame: object) -> None:
        """Publish *frame* to the attached bus.

        If no bus has been attached, this method is a no-op.

        Args:
            frame: The dataclass instance to broadcast.
        """
        if self._bus is not None:
            self._bus.publish(frame)
