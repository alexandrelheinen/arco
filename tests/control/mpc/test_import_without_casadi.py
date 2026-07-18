"""CasADi-optional import contract for arco.control.mpc."""

from __future__ import annotations

import builtins
import sys

import pytest


def test_import_without_casadi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package imports without CasADi; constructing the solver raises."""
    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "casadi" or name.startswith("casadi."):
            raise ImportError("No module named 'casadi'")
        return real_import(name, globals, locals, fromlist, level)

    # Drop cached casadi / mpc solver modules so the block is effective.
    for key in list(sys.modules):
        if key == "casadi" or key.startswith("casadi."):
            monkeypatch.delitem(sys.modules, key, raising=False)
        if key.startswith("arco.control.mpc.path_following"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    import arco.control.mpc as mpc_pkg

    assert hasattr(mpc_pkg, "ReferencePath")
    assert hasattr(mpc_pkg, "MPCStepResult")

    from arco.control.mpc.path_following import (
        DubinsPathFollowingMPC,
        DubinsVehicleLimits,
        PathFollowingMPCConfig,
        _require_casadi,
    )

    with pytest.raises(ImportError, match="pip install arco\\[mpc\\]"):
        _require_casadi()

    with pytest.raises(ImportError, match="pip install arco\\[mpc\\]"):
        DubinsPathFollowingMPC(
            vehicle_limits=DubinsVehicleLimits(
                max_speed=1.0,
                min_speed=0.0,
                max_turn_rate=1.0,
                max_acceleration=1.0,
                max_turn_rate_dot=1.0,
            ),
            config=PathFollowingMPCConfig(),
        )
