"""PipelineRunner: wires pipeline nodes to a shared bus and manages their lifecycle.

The :class:`PipelineRunner` is the single entry point for the async
pipeline.  It:

1. Reads the YAML configuration file supplied at construction time.
2. Owns an :class:`~arco.middleware.bus.InMemoryBus` instance.
3. Accepts :class:`~arco.pipeline.node.PipelineNode` instances via
   :meth:`register_node`.
4. Starts all nodes in dependency order when :meth:`start` is called.
5. Allows :class:`~arco.middleware.subscriber.BusSubscriber` frontends
   to be attached at any time — before **or** after :meth:`start` (late
   subscriber support).
6. Stops all nodes gracefully when :meth:`stop` is called.

Example::

    runner = PipelineRunner("config/map.yml")
    runner.register_node(mapping_node)
    runner.register_node(planning_node)
    runner.register_node(guidance_node)
    runner.start()

    # Frontend can be attached later:
    frontend = ArcoExFrontend()
    runner.attach_subscriber(frontend, GuidanceFrame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from arco.middleware.bus import InMemoryBus
from arco.middleware.subscriber import BusSubscriber
from arco.pipeline.node import PipelineNode

T = TypeVar("T")


class PipelineRunner:
    """Orchestrates the async pipeline from a YAML configuration file.

    Args:
        config_path: Path to the ``.yml`` map / pipeline configuration
            file.  The contents are loaded into :attr:`config` and made
            available to registered nodes.
        bus_maxsize: Maximum per-subscriber queue depth.  Frames are
            dropped silently when a consumer queue is full.  Defaults
            to ``64``.
    """

    def __init__(
        self,
        config_path: str | Path,
        bus_maxsize: int = 64,
    ) -> None:
        """Initialize the runner by loading *config_path*.

        Args:
            config_path: Path to the pipeline YAML configuration file.
            bus_maxsize: Per-subscriber queue capacity passed to
                :class:`~arco.middleware.bus.InMemoryBus`.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            RuntimeError: If the ``pyyaml`` library is not installed.
        """
        self._config_path: Path = Path(config_path)
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {self._config_path}"
            )
        self._config: Dict[str, Any] = self._load_config(self._config_path)
        self._bus: InMemoryBus = InMemoryBus(maxsize=bus_maxsize)
        self._nodes: List[PipelineNode] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        """The parsed YAML configuration dictionary.

        Returns:
            A dictionary with the contents of the configuration file.
        """
        return self._config

    @property
    def bus(self) -> InMemoryBus:
        """The shared in-memory bus owned by this runner.

        Returns:
            The :class:`~arco.middleware.bus.InMemoryBus` instance.
        """
        return self._bus

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register_node(self, node: PipelineNode) -> None:
        """Register a pipeline node with this runner.

        The node's bus is wired to the runner's shared bus.  If the
        runner has already been started, the node is also started
        immediately.

        Args:
            node: The :class:`~arco.pipeline.node.PipelineNode` to
                register.
        """
        node.attach_bus(self._bus)
        self._nodes.append(node)

    # ------------------------------------------------------------------
    # Subscriber management (late-subscriber support)
    # ------------------------------------------------------------------

    def attach_subscriber(
        self,
        subscriber: BusSubscriber,
        frame_type: Type[T],
    ) -> None:
        """Register a frontend subscriber for *frame_type*.

        This method is safe to call before or after :meth:`start`.  The
        subscriber will receive all frames published **after** this call.

        Args:
            subscriber: The :class:`~arco.middleware.subscriber.BusSubscriber`
                frontend to register.
            frame_type: The dataclass type the subscriber wants to receive.
        """
        subscriber.subscribe(self._bus, frame_type)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all registered pipeline nodes.

        Nodes are started in the order they were registered.  Nodes
        that are already running are skipped.
        """
        for node in self._nodes:
            if not node.is_running:
                node.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop all registered pipeline nodes.

        Signals every node to stop and waits for each to finish.  Nodes
        are stopped in reverse registration order.

        Args:
            timeout: Per-node join timeout in seconds.  Defaults to
                ``None`` (wait indefinitely per node).
        """
        for node in reversed(self._nodes):
            node.stop(timeout=timeout)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        """Parse *path* as YAML and return the resulting dictionary.

        Args:
            path: The configuration file to load.

        Returns:
            Parsed configuration as a dictionary.

        Raises:
            RuntimeError: If ``pyyaml`` is not installed.
        """
        if yaml is None:  # pragma: no cover
            raise RuntimeError(
                "pyyaml is required to load pipeline configuration files. "
                "Install it with: pip install pyyaml"
            )
        text = path.read_text(encoding="utf-8")
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
