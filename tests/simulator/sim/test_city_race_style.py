"""Tests for city race style constants and MPC horizon wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

from arco.control.mpc import PathFollowingMPCConfig
from arco.simulator.sim.city_race_style import (
    DEFAULT_CITY_HORIZON_DT,
    DEFAULT_CITY_HORIZON_STEP_COUNT,
    LOOKAHEAD_DISC_R,
    PAST_TRACE_WIDTH,
    PREDICTED_TRACE_WIDTH,
    VEH_HALF_L,
    VEH_HALF_W,
)
from arco.simulator.sim.tracking import (
    path_following_mpc_config_from_simulator,
)


def test_city_race_vehicle_is_double_prior_glyph() -> None:
    # Prior city glyph was 3.0 × 1.4 m half-extents; user asked for 2×.
    assert VEH_HALF_L == 6.0
    assert VEH_HALF_W == 2.8
    assert LOOKAHEAD_DISC_R >= 12.0


def test_city_race_traces_are_thicker_and_prediction_visible() -> None:
    assert PAST_TRACE_WIDTH >= 3.0
    assert PREDICTED_TRACE_WIDTH >= 4.0


def test_city_default_horizon_is_six_seconds() -> None:
    assert DEFAULT_CITY_HORIZON_STEP_COUNT == 120
    assert abs(DEFAULT_CITY_HORIZON_DT - 0.05) < 1e-12
    assert DEFAULT_CITY_HORIZON_STEP_COUNT * DEFAULT_CITY_HORIZON_DT == 6.0


def test_path_following_mpc_config_uses_city_defaults_without_yaml() -> None:
    global_cfg = PathFollowingMPCConfig.create_from_config()
    assert global_cfg.horizon_step_count == 20
    city_cfg = path_following_mpc_config_from_simulator(
        {"tracker": "mpc"},
        default_horizon_step_count=DEFAULT_CITY_HORIZON_STEP_COUNT,
        default_horizon_dt=DEFAULT_CITY_HORIZON_DT,
    )
    assert city_cfg.horizon_step_count == 120
    assert abs(city_cfg.dt - 0.05) < 1e-12


def test_path_following_mpc_config_honors_yaml_override() -> None:
    cfg = path_following_mpc_config_from_simulator(
        {
            "tracker": "mpc",
            "mpc": {"horizon": {"step_count": 48, "dt": 0.04}},
        },
        default_horizon_step_count=120,
        default_horizon_dt=0.05,
    )
    assert cfg.horizon_step_count == 48
    assert abs(cfg.dt - 0.04) < 1e-12


def test_city_map_yaml_declares_long_horizon() -> None:
    root = Path(__file__).resolve().parents[3]
    for name in ("city.yml", "city_mpc_preview.yml"):
        data = yaml.safe_load((root / "map" / name).read_text())
        horizon = data["simulator"]["mpc"]["horizon"]
        assert int(horizon["step_count"]) >= 120
        assert float(horizon["dt"]) * int(horizon["step_count"]) >= 6.0
