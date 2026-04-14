"""Bus: abstract shared in-memory message bus and InMemoryBus implementation.

The bus is the backbone of the async pipeline.  Every pipeline node that
produces output calls :meth:`Bus.publish`; every consumer (frontend,
monitor) calls :meth:`Bus.subscribe` to obtain a :class:`queue.Queue`
that receives frames of a specific type.

Design decisions:
- One :class:`queue.Queue` per ``(frame_type, subscriber)`` pair.  This
  lets multiple independent consumers each receive every published frame.
- Queues are bounded (``maxsize``).  When a consumer falls behind and its
  queue is full, :meth:`InMemoryBus.publish` silently drops the frame via
  :meth:`queue.Queue.put_nowait`.  This ensures the producer is never
  blocked by a slow consumer.
- Thread-safe: a :class:`threading.Lock` guards the subscriber registry.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Type, TypeVar

T = TypeVar("T")

_DEFAULT_MAXSIZE: int = 64


class Bus(ABC):
    """Abstract base for the shared in-memory message bus.

    Subclasses must implement :meth:`publish` and :meth:`subscribe`.
    """

    @abstractmethod
    def publish(self, frame: object) -> None:
        """Publish *frame* to all subscribers registered for its type.

        Args:
            frame: The dataclass instance to broadcast.  The frame's
                runtime type (``type(frame)``) is used to route it to
                the correct subscriber queues.
        """

    @abstractmethod
    def subscribe(self, frame_type: Type[T]) -> "queue.Queue[T]":
        """Register a new subscriber queue for *frame_type*.

        Each call returns a **new**, independent queue.  Multiple calls
        with the same *frame_type* produce separate queues so that
        multiple consumers can each receive every published frame.

        Subscribers may be registered at any time — before, during, or
        after the pipeline has started (late-subscriber support).

        Args:
            frame_type: The dataclass type to subscribe to.

        Returns:
            A :class:`queue.Queue` that will receive frames of
            *frame_type* published after this call.
        """

    @abstractmethod
    def subscriber_count(self, frame_type: Type[T]) -> int:
        """Return the number of active subscribers for *frame_type*.

        Args:
            frame_type: The dataclass type to query.

        Returns:
            Number of subscriber queues currently registered for
            *frame_type*.
        """


class InMemoryBus(Bus):
    """Thread-safe in-process bus backed by bounded :class:`queue.Queue` instances.

    Frames are routed by runtime type.  If a consumer's queue is full
    when :meth:`publish` is called, the frame is silently dropped so that
    slow consumers never block the producer.

    Args:
        maxsize: Maximum capacity of each per-subscriber queue.  Defaults
            to ``64``.  A value of ``0`` creates unbounded queues (use
            with caution).
    """

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        """Initialize the bus with optional queue capacity.

        Args:
            maxsize: Maximum number of frames each subscriber queue can
                hold before frames are dropped.
        """
        self._maxsize: int = maxsize
        self._queues: dict[type, List[queue.Queue]] = {}
        self._lock: threading.Lock = threading.Lock()

    def publish(self, frame: object) -> None:
        """Publish *frame* to all registered subscribers for its type.

        Frames are delivered via :meth:`queue.Queue.put_nowait`.  If a
        subscriber queue is full, the frame is dropped silently — the
        producer is never blocked.

        Args:
            frame: The dataclass instance to broadcast.
        """
        frame_type = type(frame)
        with self._lock:
            queues = list(self._queues.get(frame_type, []))
        for q in queues:
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass  # Slow consumer; frame dropped intentionally.

    def subscribe(self, frame_type: Type[T]) -> "queue.Queue[T]":
        """Register and return a new subscriber queue for *frame_type*.

        This method is safe to call at any time, including after the
        pipeline has started (late-subscriber support).

        Args:
            frame_type: The dataclass type to subscribe to.

        Returns:
            A new :class:`queue.Queue` that will receive all
            subsequently published frames of *frame_type*.
        """
        q: queue.Queue[T] = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            if frame_type not in self._queues:
                self._queues[frame_type] = []
            self._queues[frame_type].append(q)
        return q

    def subscriber_count(self, frame_type: Type[T]) -> int:
        """Return the number of subscriber queues for *frame_type*.

        Args:
            frame_type: The dataclass type to query.

        Returns:
            Number of subscriber queues currently registered.
        """
        with self._lock:
            return len(self._queues.get(frame_type, []))

    def unsubscribe(self, frame_type: Type[T], q: "queue.Queue[T]") -> None:
        """Remove a previously registered subscriber queue.

        If *q* is not registered for *frame_type*, the call is a no-op.

        Args:
            frame_type: The dataclass type the queue was registered for.
            q: The queue to remove.
        """
        with self._lock:
            queues = self._queues.get(frame_type, [])
            if q in queues:
                queues.remove(q)
