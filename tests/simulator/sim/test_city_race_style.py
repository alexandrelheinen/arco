"""Tests for city race style constants and MPC horizon wiring."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from arco.config import load_config
from arco.control.mpc import PathFollowingMPCConfig
from arco.simulator.sim.city_race_style import (
    CITY_CRUISE_SPEED,
    CITY_MAX_ACCELERATION,
    CITY_MAX_SPEED,
    CITY_MAX_TURN_RATE_DEG,
    CITY_MAX_TURN_RATE_DOT_DEG,
    CITY_ROAD_HALF_WIDTH,
    DEFAULT_CITY_HORIZON_DT,
    DEFAULT_CITY_HORIZON_STEP_COUNT,
    LOOKAHEAD_DISC_R,
    PAST_TRACE_WIDTH,
    PLANNED_ROUTE_ALPHA,
    PLANNED_ROUTE_WIDTH,
    PREDICTED_TRACE_WIDTH,
    VEH_HALF_L,
    VEH_HALF_W,
    make_city_vehicle_config,
)
from arco.simulator.sim.tracking import (
    path_following_mpc_config_from_simulator,
)


def test_city_race_vehicle_is_visible_rectangle() -> None:
    # Visible car glyph (12 × 5.4 m), but small enough that the sprite's
    # nose cannot sweep over inner-corner buildings on routes that pass
    # ~7.7 m from them: tip reach (half-length) must stay well below the
    # minimum legitimate center clearance.
    assert VEH_HALF_L == 6.0
    assert VEH_HALF_W == 2.7
    assert VEH_HALF_L <= 7.0
    # Lookahead disc is only a small PP carrot, never the vehicle body.
    assert LOOKAHEAD_DISC_R <= 2.0
    assert LOOKAHEAD_DISC_R < min(VEH_HALF_L, VEH_HALF_W)


def test_city_race_traces_are_thicker_and_prediction_visible() -> None:
    # Executed past trail is the visual hero over a dim planned underlay.
    assert PAST_TRACE_WIDTH >= 4.0
    assert PREDICTED_TRACE_WIDTH >= 3.0
    assert PREDICTED_TRACE_WIDTH < PAST_TRACE_WIDTH
    assert PLANNED_ROUTE_WIDTH < PAST_TRACE_WIDTH
    assert 0.0 < PLANNED_ROUTE_ALPHA < 0.6


def test_city_default_horizon_matches_simulator_timestep() -> None:
    """City MPC model dt must equal the 0.1 s simulator timestep.

    The MPC's first predicted state is the command target; a model dt
    shorter than the control period makes the plant travel further per
    tick than planned (the historical race zigzag).  40 × 0.1 s = 4.0 s
    of preview (~48 m at 12 m/s cruise) covers corner braking distance.
    """
    sim_cfg = load_config("simulator")
    assert abs(DEFAULT_CITY_HORIZON_DT - float(sim_cfg["timestep"])) < 1e-12
    assert DEFAULT_CITY_HORIZON_STEP_COUNT == 50
    # Preview must exceed the full-stop braking time from cruise.
    preview_s = DEFAULT_CITY_HORIZON_STEP_COUNT * DEFAULT_CITY_HORIZON_DT
    assert preview_s > CITY_CRUISE_SPEED / CITY_MAX_ACCELERATION


def test_city_vehicle_dynamics_are_soft_but_lane_viable() -> None:
    """City racers stay soft, but curve-limited corners fit the road.

    Cruise ``R_min`` may still exceed half-width (visible understeer if the
    car does not slow), but a 90° / 15 m A* grid corner yields
    ``R = 1/κ < road_half_width`` once ``v_curve = ω/|κ|`` engages.
    """
    assert CITY_CRUISE_SPEED == 12.0
    assert CITY_MAX_SPEED == 16.0
    assert CITY_MAX_TURN_RATE_DEG == 40.0
    assert CITY_MAX_ACCELERATION == 2.5
    assert CITY_MAX_TURN_RATE_DOT_DEG == 90.0
    assert CITY_ROAD_HALF_WIDTH == 15.0
    cfg = make_city_vehicle_config()
    assert abs(cfg.cruise_speed - CITY_CRUISE_SPEED) < 1e-12
    assert abs(cfg.max_turn_rate - math.radians(40.0)) < 1e-12
    assert abs(cfg.max_turn_rate_dot - math.radians(90.0)) < 1e-12
    assert abs(cfg.max_acceleration - 2.5) < 1e-12
    # Soft: still cannot snap-turn a kink at full cruise.
    assert cfg.cruise_speed / cfg.max_turn_rate > CITY_ROAD_HALF_WIDTH
    # Lane-viable at a grid corner once curve speed limiting engages.
    kappa_grid_corner = (math.pi / 2.0) / 15.0
    r_corner = 1.0 / kappa_grid_corner
    assert r_corner < CITY_ROAD_HALF_WIDTH


def test_path_following_mpc_config_uses_city_defaults_without_yaml() -> None:
    global_cfg = PathFollowingMPCConfig.create_from_config()
    assert global_cfg.horizon_step_count == 20
    city_cfg = path_following_mpc_config_from_simulator(
        {"tracker": "mpc"},
        default_horizon_step_count=DEFAULT_CITY_HORIZON_STEP_COUNT,
        default_horizon_dt=DEFAULT_CITY_HORIZON_DT,
    )
    assert city_cfg.horizon_step_count == 50
    assert abs(city_cfg.dt - 0.1) < 1e-12


def test_path_following_mpc_config_honors_yaml_override() -> None:
    cfg = path_following_mpc_config_from_simulator(
        {
            "tracker": "mpc",
            "mpc": {"horizon": {"step_count": 48, "dt": 0.04}},
        },
        default_horizon_step_count=72,
        default_horizon_dt=0.05,
    )
    assert cfg.horizon_step_count == 48
    assert abs(cfg.dt - 0.04) < 1e-12


def test_path_following_mpc_config_honors_weight_overrides() -> None:
    cfg = path_following_mpc_config_from_simulator(
        {
            "tracker": "mpc",
            "mpc": {
                "horizon": {"step_count": 72, "dt": 0.05},
                "weights": {
                    "contour": 8.0,
                    "heading": 4.0,
                    "control": 0.5,
                    "progress": 4.0,
                    "lag": 4.0,
                    "contour_deadzone": 0.0,
                },
            },
        },
        default_horizon_step_count=72,
        default_horizon_dt=0.05,
    )
    assert abs(cfg.weight_contour - 8.0) < 1e-12
    assert abs(cfg.weight_heading - 4.0) < 1e-12
    assert abs(cfg.weight_control - 0.5) < 1e-12
    assert abs(cfg.weight_progress - 4.0) < 1e-12
    assert abs(cfg.weight_lag - 4.0) < 1e-12
    assert abs(cfg.contour_deadzone) < 1e-12


def test_city_map_yaml_declares_mpcc_tuning() -> None:
    """City YAML: horizon dt == sim timestep, MPCC weights sane."""
    root = Path(__file__).resolve().parents[3]
    sim_cfg = load_config("simulator")
    for name in ("city.yml", "city_mpc_preview.yml"):
        data = yaml.safe_load((root / "map" / name).read_text())
        horizon = data["simulator"]["mpc"]["horizon"]
        # Model step must match the closed-loop control period.
        assert abs(float(horizon["dt"]) - float(sim_cfg["timestep"])) < 1e-12
        # Enough preview to brake from cruise before a sharp corner.
        preview_s = float(horizon["dt"]) * int(horizon["step_count"])
        braking_s = CITY_CRUISE_SPEED / CITY_MAX_ACCELERATION
        assert preview_s > braking_s
        weights = data["simulator"]["mpc"]["weights"]
        # Contour dominates; lag is structural (couples s to the vehicle).
        assert float(weights["contour"]) >= 8.0
        assert float(weights["lag"]) >= 4.0
        assert float(weights["progress"]) > 0.0
        # Heading stays a light alignment term (tracking emerges from
        # contour + lag; heavy heading tracking fights kinked references).
        assert float(weights["heading"]) <= 2.0
        # Flat |e_lat| bands zero the lateral gradient → equal-cost chatter.
        assert abs(float(weights["contour_deadzone"])) < 1e-12
        assert float(weights["obstacle"]) >= 100.0
