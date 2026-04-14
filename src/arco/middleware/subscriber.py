"""BusSubscriber: mixin that equips a frontend node with bus subscription.

Import and inherit from :class:`BusSubscriber` to give any class the
ability to subscribe to typed frames on the shared in-memory bus and
consume them at its own pace.

Example::

    class ArcoExFrontend(BusSubscriber):
        def attach(self, bus: Bus) -> None:
            self._guidance_q = self.subscribe(bus, GuidanceFrame)

        def render(self) -> None:
            frame = self.next_frame(GuidanceFrame)
            if frame is not None:
                ...  # render the frame
"""

from __future__ import annotations

import queue
from typing import Optional, Type, TypeVar

from .bus import Bus

T = TypeVar("T")


class BusSubscriber:
    """Mixin that adds bus-subscription capability to a frontend node.

    Call :meth:`subscribe` to register a per-type queue on a
    :class:`~arco.middleware.bus.Bus` instance.  Then call
    :meth:`next_frame` in the render loop to dequeue the latest frame
    without blocking the pipeline.

    Subscriptions can be created at any time — the bus supports
    late-subscriber registration.
    """

    def __init__(self) -> None:
        """Initialize the subscriber with an empty subscription registry."""
        self._subscriptions: dict[type, queue.Queue] = {}

    def subscribe(self, bus: Bus, frame_type: Type[T]) -> "queue.Queue[T]":
        """Register a new subscriber queue for *frame_type* on *bus*.

        The returned queue is also stored internally so that
        :meth:`next_frame` can retrieve frames by type.

        Args:
            bus: The shared :class:`~arco.middleware.bus.Bus` to
                subscribe on.
            frame_type: The dataclass type to subscribe to.

        Returns:
            The newly created :class:`queue.Queue` for *frame_type*.
        """
        q = bus.subscribe(frame_type)
        self._subscriptions[frame_type] = q
        return q

    def next_frame(
        self,
        frame_type: Type[T],
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[T]:
        """Retrieve the next available frame of *frame_type*.

        This method never raises :class:`queue.Empty` — it returns
        ``None`` instead, so the caller can safely poll in a render loop
        without try/except boilerplate.

        Args:
            frame_type: The dataclass type to dequeue.
            block: If ``True``, block until a frame is available or
                *timeout* expires.  Defaults to ``False`` (non-blocking
                poll).
            timeout: Maximum seconds to wait when *block* is ``True``.
                Ignored when *block* is ``False``.

        Returns:
            The next frame, or ``None`` if the queue is empty (or the
            type has not been subscribed to).
        """
        q = self._subscriptions.get(frame_type)
        if q is None:
            return None
        try:
            return q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def drain_latest(self, frame_type: Type[T]) -> Optional[T]:
        """Drain the queue and return only the most recent frame.

        Useful for frontends that want to skip intermediate frames when
        they fall behind the pipeline (i.e. drop all but the latest).

        Args:
            frame_type: The dataclass type to drain.

        Returns:
            The most recent frame available, or ``None`` if the queue
            is empty or the type is not subscribed.
        """
        q = self._subscriptions.get(frame_type)
        if q is None:
            return None
        latest: Optional[T] = None
        while True:
            try:
                latest = q.get_nowait()
            except queue.Empty:
                break
        return latest
