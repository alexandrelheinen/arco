"""Shared simulator engine for ARCO planner visualisations."""

from __future__ import annotations

from .layout import ScreenLayout

__all__ = ["run_sim", "ScreenLayout"]


def __getattr__(name: str) -> object:
    """Lazily import run_sim to avoid eagerly loading pygame."""
    if name == "run_sim":
        from .loop import run_sim  # noqa: PLC0415

        globals()["run_sim"] = run_sim
        return run_sim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
