"""Standardised JSON-serializable data model for planning results.

Both *arcoex* (static image generation) and *arcosim* (pygame simulation)
consume the same :class:`SceneSnapshot` structure, enabling a clean
middleware boundary: algorithms write a ``SceneSnapshot``, the rendering
layer reads it — no algorithm-specific rendering code needed.

The snapshot stores every visual layer described in the issue:

1. **obstacles** — static occupancy data (bounding boxes or point cloud).
2. **exploration_tree** — full search tree (nodes + parent map).
3. **found_path** — raw waypoint sequence from the planner.
4. **pruned_path** — simplified waypoints after pruning.
5. **adjusted_trajectory** — smooth optimised trajectory states.
6. **executed_trajectory** — actual recorded trace from the simulator.

Example
-------
>>> snap = SceneSnapshot(
...     scenario="rr",
...     start=[0.0, 0.0],
...     goal=[1.57, -0.5],
...     found_path=[[0.0, 0.0], [0.78, -0.25], [1.57, -0.5]],
... )
>>> import json, pathlib
>>> pathlib.Path("/tmp/snap.json").write_text(snap.to_json())
>>> snap2 = SceneSnapshot.from_json(pathlib.Path("/tmp/snap.json").read_text())
>>> snap2.scenario
'rr'
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass
class SceneSnapshot:
    """Standardised, serialisable snapshot of a single planning result.

    All coordinate sequences use lists of ``[float, ...]`` to stay JSON-
    native.  ``None`` means the layer was not produced (e.g., the planner
    found no path).

    Attributes:
        scenario: Scenario identifier string, e.g. ``"rr"`` or ``"city"``.
        planner: Planner name, e.g. ``"rrt"`` or ``"sst"``.
        start: Start state as a flat list of floats.
        goal: Goal state as a flat list of floats.
        obstacles: Obstacle representation — list of axis-aligned bounding
            boxes ``[[xmin, ymin, xmax, ymax], …]`` for 2-D, or a flat
            point cloud ``[[x, y, …], …]`` for continuous occupancy.
        tree_nodes: Exploration tree nodes, each a flat state list.
        tree_parent: Parallel index list; ``tree_parent[i]`` is the index
            of the parent of ``tree_nodes[i]``, ``-1`` for the root.
        found_path: Raw waypoint sequence from the planner (dense).
        pruned_path: Pruned / simplified waypoints (sparse).
        adjusted_trajectory: Optimised smooth trajectory states.
        executed_trajectory: Actual trace recorded during simulation.
        metrics: Freeform ``str → float/int/str`` mapping for telemetry
            (planning time, path length, optimiser status, …).
    """

    scenario: str = ""
    planner: str = ""
    start: list[float] = dataclasses.field(default_factory=list)
    goal: list[float] = dataclasses.field(default_factory=list)
    obstacles: list[list[float]] = dataclasses.field(default_factory=list)
    tree_nodes: list[list[float]] = dataclasses.field(default_factory=list)
    tree_parent: list[int] = dataclasses.field(default_factory=list)
    found_path: list[list[float]] | None = None
    pruned_path: list[list[float]] | None = None
    adjusted_trajectory: list[list[float]] | None = None
    executed_trajectory: list[list[float]] | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert this snapshot to a plain JSON-serialisable dictionary.

        Returns:
            A ``dict`` whose values are JSON primitives (``None``, ``bool``,
            ``int``, ``float``, ``str``, or nested ``list``/``dict``).
        """
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string.

        Args:
            indent: Indentation level passed to :func:`json.dumps`.
                Use ``None`` for compact single-line output.

        Returns:
            A UTF-8 JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSnapshot:
        """Deserialise from a plain dictionary.

        Extra keys in *data* are silently ignored to allow forward
        compatibility as the schema evolves.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.

        Returns:
            A :class:`SceneSnapshot` populated from *data*.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> SceneSnapshot:
        """Deserialise from a JSON string.

        Args:
            text: A JSON string as produced by :meth:`to_json`.

        Returns:
            A :class:`SceneSnapshot` populated from the JSON data.

        Raises:
            json.JSONDecodeError: If *text* is not valid JSON.
            TypeError: If the top-level JSON value is not an object.
        """
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected a JSON object at the top level; got {type(raw)}"
            )
        return cls.from_dict(raw)

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    @classmethod
    def from_planning_result(
        cls,
        *,
        scenario: str,
        planner: str,
        start: list[float],
        goal: list[float],
        obstacles: list[list[float]] | None = None,
        tree_nodes: list[list[float]] | None = None,
        tree_parent: list[int] | None = None,
        found_path: list[list[float]] | None = None,
        pruned_path: list[list[float]] | None = None,
        adjusted_trajectory: list[list[float]] | None = None,
        executed_trajectory: list[list[float]] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> SceneSnapshot:
        """Construct a snapshot from individual planning result components.

        This is the preferred factory when calling code has already split
        data into the individual layers rather than building a dict.

        Args:
            scenario: Scenario identifier string.
            planner: Planner key (``"rrt"`` / ``"sst"`` / …).
            start: Start state floats.
            goal: Goal state floats.
            obstacles: Obstacle data; defaults to an empty list.
            tree_nodes: Exploration tree nodes; defaults to empty.
            tree_parent: Parent indices parallel to *tree_nodes*; defaults
                to empty.
            found_path: Raw planner path; ``None`` if not found.
            pruned_path: Pruned path; ``None`` if not pruned.
            adjusted_trajectory: Optimised trajectory; ``None`` if not
                computed.
            executed_trajectory: Executed trace; ``None`` if not recorded.
            metrics: Telemetry key-value pairs; defaults to empty dict.

        Returns:
            A fully-populated :class:`SceneSnapshot`.
        """
        return cls(
            scenario=scenario,
            planner=planner,
            start=start,
            goal=goal,
            obstacles=obstacles or [],
            tree_nodes=tree_nodes or [],
            tree_parent=tree_parent or [],
            found_path=found_path,
            pruned_path=pruned_path,
            adjusted_trajectory=adjusted_trajectory,
            executed_trajectory=executed_trajectory,
            metrics=metrics or {},
        )
