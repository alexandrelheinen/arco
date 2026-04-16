"""Tests for the ArcosimScene / SimScene / RaceScene hierarchy."""

from __future__ import annotations

import pytest

pygame = pytest.importorskip("pygame")


def test_arcosimscene_importable() -> None:
    from arco.tools.simulator.sim.scene import (
        ArcosimScene,
        RaceScene,
        SimScene,
    )

    assert issubclass(SimScene, ArcosimScene)
    assert issubclass(RaceScene, ArcosimScene)


def test_rrtscene_is_simscene() -> None:
    from arco.tools.simulator.scenes.rrt import RRTScene
    from arco.tools.simulator.sim.scene import ArcosimScene, SimScene

    assert issubclass(RRTScene, SimScene)
    assert issubclass(RRTScene, ArcosimScene)


def test_sstscene_is_simscene() -> None:
    from arco.tools.simulator.scenes.sst import SSTScene
    from arco.tools.simulator.sim.scene import ArcosimScene, SimScene

    assert issubclass(SSTScene, SimScene)
    assert issubclass(SSTScene, ArcosimScene)


def test_astarscene_is_simscene() -> None:
    from arco.tools.simulator.scenes.astar import AStarScene
    from arco.tools.simulator.sim.scene import ArcosimScene, SimScene

    assert issubclass(AStarScene, SimScene)
    assert issubclass(AStarScene, ArcosimScene)


def test_cityscene_is_racescene() -> None:
    from arco.tools.simulator.scenes.sparse import CityScene
    from arco.tools.simulator.sim.scene import ArcosimScene, RaceScene

    assert issubclass(CityScene, RaceScene)
    assert issubclass(CityScene, ArcosimScene)


def test_vehiclescene_is_racescene() -> None:
    from arco.tools.simulator.scenes.vehicle import VehicleScene
    from arco.tools.simulator.sim.scene import ArcosimScene, RaceScene

    assert issubclass(VehicleScene, RaceScene)
    assert issubclass(VehicleScene, ArcosimScene)


def test_footer_hint_default() -> None:
    from arco.tools.simulator.sim.scene import ArcosimScene

    class _Stub(ArcosimScene):
        def build(self, *, progress=None) -> None:
            pass

        @property
        def title(self) -> str:
            return "stub"

        @property
        def bg_color(self) -> tuple[int, int, int]:
            return (0, 0, 0)

        @property
        def world_points(self) -> list[tuple[float, float]]:
            return []

        def sidebar_content(self, **state):
            return []

    stub = _Stub()
    assert "SPACE" in stub.footer_hint
    assert "pause" in stub.footer_hint


def test_background_total_defaults_to_zero() -> None:
    from arco.tools.simulator.sim.scene import ArcosimScene

    class _Stub(ArcosimScene):
        def build(self, *, progress=None) -> None:
            pass

        @property
        def title(self) -> str:
            return "stub"

        @property
        def bg_color(self) -> tuple[int, int, int]:
            return (0, 0, 0)

        @property
        def world_points(self) -> list[tuple[float, float]]:
            return []

        def sidebar_content(self, **state):
            return []

    assert _Stub().background_total == 0


def test_scenes_exported_from_scenes_init() -> None:
    from arco.tools.simulator.scenes import ArcosimScene, RaceScene, SimScene

    assert ArcosimScene is not None
    assert SimScene is not None
    assert RaceScene is not None
