from arco.guidance.mpc import MPCController
from arco.guidance.pid import PIDController
from arco.guidance.pure_pursuit import PurePursuitController


def test_pure_pursuit_controller():
    ctrl = PurePursuitController(lookahead_distance=2.0)
    cmd = ctrl.control(0.0, 1.0)
    assert isinstance(cmd, float)


def test_pid_controller():
    ctrl = PIDController(kp=1.0, ki=0.5, kd=0.1)
    cmd1 = ctrl.control(0.0, 1.0)
    cmd2 = ctrl.control(0.5, 1.0)
    assert isinstance(cmd1, float)
    assert isinstance(cmd2, float)


def test_mpc_controller():
    ctrl = MPCController(horizon=5, dt=0.2)
    cmd = ctrl.control(0.0, 1.0)
    assert isinstance(cmd, float)
