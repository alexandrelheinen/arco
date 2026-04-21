"""Planner-specific scenes for the ARCO unified simulator."""

from __future__ import annotations

__all__ = ["ArcosimScene", "RaceScene", "SimScene"]


def __getattr__(name: str) -> object:
    """Lazily import hierarchy classes to avoid eagerly loading pygame."""
    if name in __all__:
        from arco.simulator.sim.scene import (  # noqa: PLC0415
            ArcosimScene,
            RaceScene,
            SimScene,
        )

        g = globals()
        g["ArcosimScene"] = ArcosimScene
        g["RaceScene"] = RaceScene
        g["SimScene"] = SimScene
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
