"""PID feedback controller, re-exported from the compiled extension.

The implementation is ``PidController`` in the ``arco-control`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`. Deviation A-18: the Rust
controller takes the elapsed interval, and the binding passes exactly one
second, so the arithmetic matches the Python controller apart from the
first step after construction or reset, which takes no derivative.
Deviation A-09: saturation, rate limiting and anti-windup are available on
the crate and contribute nothing at the default settings the binding uses.
"""

from arco._arco import PIDController

__all__ = ["PIDController"]
