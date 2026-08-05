"""Joint-space MPC horizon must match scenario control period (PPP / RRP)."""

from __future__ import annotations

from pathlib import Path

import yaml

from arco.config import load_config
from arco.simulator.sim.tracking import (
    DEFAULT_JOINT_HORIZON_DT,
    DEFAULT_JOINT_HORIZON_STEP_COUNT,
    joint_space_mpc_config_from_simulator,
    resolve_sim_timestep,
)


def test_joint_default_horizon_matches_ppp_timestep() -> None:
    """PPP uses 0.05 s steps; joint MPC model dt must match."""
    global_sim = load_config("simulator")
    ppp = yaml.safe_load(Path("map/ppp.yml").read_text(encoding="utf-8"))
    dt = resolve_sim_timestep(ppp, global_sim_cfg=global_sim)
    assert abs(dt - DEFAULT_JOINT_HORIZON_DT) < 1e-12
    assert DEFAULT_JOINT_HORIZON_STEP_COUNT == 12

    mpc_cfg = joint_space_mpc_config_from_simulator(
        ppp.get("simulator", {}),
        default_horizon_step_count=DEFAULT_JOINT_HORIZON_STEP_COUNT,
        default_horizon_dt=dt,
    )
    assert abs(mpc_cfg.dt - dt) < 1e-12
    assert mpc_cfg.horizon_step_count == 12


def test_joint_default_horizon_matches_rrp_timestep() -> None:
    """RRP uses the same 0.05 s control period as PPP."""
    global_sim = load_config("simulator")
    rrp = yaml.safe_load(Path("map/rrp.yml").read_text(encoding="utf-8"))
    dt = resolve_sim_timestep(rrp, global_sim_cfg=global_sim)
    assert abs(dt - 0.05) < 1e-12

    mpc_cfg = joint_space_mpc_config_from_simulator(
        rrp.get("simulator", {}),
        default_horizon_step_count=DEFAULT_JOINT_HORIZON_STEP_COUNT,
        default_horizon_dt=dt,
    )
    assert abs(mpc_cfg.dt - dt) < 1e-12


def test_map_yaml_joint_mpc_horizon_matches_timestep() -> None:
    """Scenario YAML must declare mpc.horizon.dt == simulator.timestep."""
    global_sim = load_config("simulator")
    for name in ("ppp.yml", "rrp.yml"):
        data = yaml.safe_load(Path(f"map/{name}").read_text(encoding="utf-8"))
        sim = data["simulator"]
        dt = resolve_sim_timestep(data, global_sim_cfg=global_sim)
        horizon = sim["mpc"]["horizon"]
        assert abs(float(horizon["dt"]) - dt) < 1e-12
        assert int(horizon["step_count"]) > 0


def test_resolve_sim_timestep_falls_back_to_global() -> None:
    """Scenarios without an override use simulator.yml timestep."""
    global_sim = load_config("simulator")
    city = yaml.safe_load(Path("map/city.yml").read_text(encoding="utf-8"))
    dt = resolve_sim_timestep(city, global_sim_cfg=global_sim)
    assert abs(dt - float(global_sim["timestep"])) < 1e-12


def test_occ_timestep_matches_global() -> None:
    """OCC has no MPC but should resolve the global 0.1 s period."""
    global_sim = load_config("simulator")
    occ = yaml.safe_load(Path("map/occ.yml").read_text(encoding="utf-8"))
    dt = resolve_sim_timestep(occ, global_sim_cfg=global_sim)
    assert abs(dt - float(global_sim["timestep"])) < 1e-12
