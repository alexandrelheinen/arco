"""Unit tests for PPPRobot class."""

from __future__ import annotations

import numpy as np
import pytest

from arco.guidance.ppp import PPPRobot


def test_initial_state():
    """Test initial state is set correctly."""
    robot = PPPRobot(x=1.0, y=2.0, z=3.0)
    assert robot.x == 1.0
    assert robot.y == 2.0
    assert robot.z == 3.0
    assert np.allclose(robot.position, [1.0, 2.0, 3.0])
    assert np.allclose(robot.velocity, [0.0, 0.0, 0.0])


def test_position_property():
    """Test position property returns numpy array."""
    robot = PPPRobot(x=5.0, y=7.0, z=2.0)
    pos = robot.position
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert np.allclose(pos, [5.0, 7.0, 2.0])


def test_velocity_property():
    """Test velocity property returns numpy array."""
    robot = PPPRobot()
    vel = robot.velocity
    assert isinstance(vel, np.ndarray)
    assert vel.shape == (3,)
    assert np.allclose(vel, [0.0, 0.0, 0.0])


def test_step_returns_position_array():
    """Test step returns position as numpy array."""
    robot = PPPRobot()
    pos = robot.step(0.0, 0.0, 0.0, 0.1)
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)


def test_acceleration_integration():
    """Test that acceleration integrates to velocity."""
    robot = PPPRobot(max_acceleration=1.0, max_velocity=10.0)
    # Apply 1 m/s² for 1 second → velocity should be 1 m/s
    robot.step(1.0, 0.0, 0.0, 1.0)
    assert np.isclose(robot.velocity[0], 1.0)
    assert np.isclose(robot.velocity[1], 0.0)
    assert np.isclose(robot.velocity[2], 0.0)


def test_velocity_integration():
    """Test that velocity integrates to position."""
    robot = PPPRobot(max_acceleration=1.0, max_velocity=10.0)
    # Apply 1 m/s² for 1 second → velocity = 1 m/s, position += 1 m
    robot.step(1.0, 0.0, 0.0, 1.0)
    # Apply 0 m/s² for 1 second → velocity stays 1 m/s, position += 1 m
    robot.step(0.0, 0.0, 0.0, 1.0)
    # Total displacement: 1 m (first step) + 1 m (second step) = 2 m
    assert np.isclose(robot.x, 2.0, atol=1e-6)


def test_velocity_saturation():
    """Test that velocity saturates at max_velocity."""
    robot = PPPRobot(max_acceleration=2.0, max_velocity=1.0)
    # Apply large acceleration for long time
    robot.step(10.0, 10.0, 10.0, 2.0)
    # Velocity should be clamped to max_velocity
    assert np.allclose(robot.velocity, [1.0, 1.0, 1.0])


def test_acceleration_saturation():
    """Test that acceleration saturates at max_acceleration."""
    robot = PPPRobot(max_acceleration=0.5, max_velocity=10.0)
    # Command large acceleration
    robot.step(100.0, 100.0, 100.0, 1.0)
    # Velocity should only increase by max_acceleration * dt
    assert np.allclose(robot.velocity, [0.5, 0.5, 0.5])


def test_independent_axis_control():
    """Test that each axis is controlled independently."""
    robot = PPPRobot(max_acceleration=1.0, max_velocity=10.0)
    # Apply different accelerations to each axis
    robot.step(1.0, 0.5, 0.2, 1.0)
    assert np.isclose(robot.velocity[0], 1.0)
    assert np.isclose(robot.velocity[1], 0.5)
    assert np.isclose(robot.velocity[2], 0.2)


def test_workspace_constraint_x():
    """Test that robot is constrained to workspace x bounds."""
    robot = PPPRobot(
        x=5.0,
        work_volume=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        max_acceleration=10.0,
        max_velocity=100.0,
    )
    # Try to move far beyond workspace
    for _ in range(20):
        robot.step(10.0, 0.0, 0.0, 1.0)
    # Should be clamped to max x
    assert robot.x <= 10.0


def test_workspace_constraint_z():
    """Test that robot is constrained to workspace z bounds."""
    robot = PPPRobot(
        z=5.0,
        work_volume=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        max_acceleration=10.0,
        max_velocity=100.0,
    )
    # Try to move below workspace
    for _ in range(20):
        robot.step(0.0, 0.0, -10.0, 1.0)
    # Should be clamped to min z
    assert robot.z >= 0.0


def test_reset_restores_state():
    """Test that reset restores position and clears velocity."""
    robot = PPPRobot(x=5.0, y=5.0, z=5.0)
    # Move the robot
    robot.step(1.0, 1.0, 1.0, 1.0)
    robot.step(1.0, 1.0, 1.0, 1.0)
    # Reset to new position
    robot.reset(x=1.0, y=2.0, z=3.0)
    assert np.allclose(robot.position, [1.0, 2.0, 3.0])
    assert np.allclose(robot.velocity, [0.0, 0.0, 0.0])


def test_default_workspace():
    """Test that default workspace is [[0,10], [0,10], [0,10]]."""
    robot = PPPRobot()
    assert robot.work_volume == [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]


def test_custom_workspace():
    """Test that custom workspace is set correctly."""
    custom = [[0.0, 5.0], [0.0, 8.0], [0.0, 12.0]]
    robot = PPPRobot(work_volume=custom)
    assert robot.work_volume == custom


def test_velocity_stops_at_boundary():
    """Test that velocity is zeroed when hitting workspace boundary."""
    robot = PPPRobot(
        x=9.5,
        work_volume=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        max_acceleration=1.0,
        max_velocity=1.0,
    )
    # Accelerate toward boundary
    robot.step(1.0, 0.0, 0.0, 1.0)
    # Step again, should hit boundary and stop
    robot.step(1.0, 0.0, 0.0, 1.0)
    assert robot.x == 10.0
    assert np.isclose(robot.velocity[0], 0.0)


def test_negative_acceleration():
    """Test that negative acceleration decelerates the robot."""
    robot = PPPRobot(max_acceleration=1.0, max_velocity=10.0)
    # Accelerate to 1 m/s
    robot.step(1.0, 0.0, 0.0, 1.0)
    # Decelerate with -0.5 m/s² for 2 seconds
    robot.step(-0.5, 0.0, 0.0, 2.0)
    # Should be at 0 m/s
    assert np.isclose(robot.velocity[0], 0.0, atol=1e-6)


def test_three_axis_motion():
    """Test simultaneous motion on all three axes."""
    robot = PPPRobot(
        x=1.0,
        y=1.0,
        z=1.0,
        max_acceleration=1.0,
        max_velocity=1.0,
    )
    # Apply same acceleration on all axes
    robot.step(1.0, 1.0, 1.0, 1.0)
    # Check all velocities are equal
    assert np.allclose(robot.velocity, [1.0, 1.0, 1.0])
    # Step again with zero acceleration
    robot.step(0.0, 0.0, 0.0, 1.0)
    # Total displacement per axis: 1 m (first) + 1 m (second) = 2 m
    # Starting from (1,1,1), final position is (3,3,3)
    assert np.allclose(robot.position, [3.0, 3.0, 3.0])
