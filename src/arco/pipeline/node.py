"""PipelineNode: abstract base class for a single pipeline stage.

Each stage runs in its own daemon thread, publishes typed dataclasses to
the shared in-memory bus, and can be started and stopped independently.
Concrete subclasses implement :meth:`run`, which is called once in the
background thread.

Example::

    class MyMappingNode(PipelineNode):
        def run(self) -> None:
            while not self.stop_requested:
                frame = MappingFrame(timestamp=time.monotonic(), ...)
                self.publish(frame)
                time.sleep(0.1)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from arco.middleware.publisher import BusPublisher


class PipelineNode(BusPublisher, ABC):
    """Abstract base class for a single stage in the async pipeline.

    A ``PipelineNode`` wraps a :class:`threading.Thread` lifecycle and
    inherits :class:`~arco.middleware.publisher.BusPublisher` so that
    it can push typed frames to the shared bus.

    Subclasses must implement :meth:`run`.  The method is invoked once
    in the background thread; long-running nodes should poll
    :attr:`stop_requested` to exit gracefully when :meth:`stop` is
    called.

    Args:
        name: Human-readable identifier for logging and debugging.
    """

    def __init__(self, name: str) -> None:
        """Initialize the node with the given name.

        Args:
            name: Human-readable label for this stage (e.g.
                ``"mapping"``, ``"planning"``, ``"guidance"``).
        """
        super().__init__()
        self._name: str = name
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

    @property
    def name(self) -> str:
        """Human-readable identifier for this node.

        Returns:
            The name supplied at construction time.
        """
        return self._name

    @property
    def stop_requested(self) -> bool:
        """True once :meth:`stop` has been called.

        Returns:
            ``True`` if the node has been asked to stop, ``False``
            otherwise.
        """
        return self._stop_event.is_set()

    def start(self) -> None:
        """Start the node's background thread.

        The thread is a daemon so that it does not prevent the process
        from exiting if the main thread finishes.  Calling :meth:`start`
        on an already-running node is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper,
            name=self._name,
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the node to stop and wait for its thread to exit.

        Sets the stop event and joins the background thread.  After this
        call returns the node may be started again with :meth:`start`.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.
                Defaults to ``None`` (wait indefinitely).
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """True if the background thread is alive.

        Returns:
            ``True`` when the node's thread is running.
        """
        return self._thread is not None and self._thread.is_alive()

    def _run_wrapper(self) -> None:
        """Internal wrapper that calls :meth:`run` and catches exceptions."""
        try:
            self.run()
        except Exception:  # noqa: BLE001
            pass

    @abstractmethod
    def run(self) -> None:
        """Execute the node's logic in the background thread.

        This method is called once.  Long-running nodes should loop
        and check :attr:`stop_requested` to exit gracefully::

            def run(self) -> None:
                while not self.stop_requested:
                    frame = ...
                    self.publish(frame)
        """
