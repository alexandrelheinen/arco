//! Planar rigid bodies, integrated forward under accumulated wrenches.

use arco_core::Error;

use crate::limits::IntervalBand;

/// The state every planar rigid body carries.
///
/// Held by composition rather than inherited, per deviation A-03: a body
/// owns one of these and the trait below reaches it, which is the same
/// arrangement `CartesianGraph` has with its topology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    mass: f64,
    pose: [f64; 3],
    velocity: [f64; 3],
    force: [f64; 2],
    torque: f64,
    interval: IntervalBand,
}

impl BodyState {
    /// Builds a state at rest at `(x, y)` facing `heading`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `mass` is not finite and
    /// strictly positive, and [`Error::NotFinite`] when a pose component
    /// is not a real number.
    pub fn new(mass: f64, x: f64, y: f64, heading: f64) -> Result<Self, Error> {
        if !(mass.is_finite() && mass > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "mass",
                value: mass,
                bound: "(0, inf)",
            });
        }
        for (quantity, value) in [("x", x), ("y", y), ("heading", heading)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        Ok(Self {
            mass,
            pose: [x, y, heading],
            velocity: [0.0; 3],
            force: [0.0; 2],
            torque: 0.0,
            interval: IntervalBand::default(),
        })
    }

    /// Mass, kilograms.
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Pose as `[x, y, heading]`, meters and radians.
    ///
    /// The heading is not wrapped. A body that has turned three times is
    /// distinguishable from one that has turned once, which a wrapped
    /// angle would lose, and the simulator draws the raw value.
    #[must_use]
    pub const fn pose(&self) -> [f64; 3] {
        self.pose
    }

    /// Velocity as `[vx, vy, turn rate]`, meters and radians per second.
    #[must_use]
    pub const fn velocity(&self) -> [f64; 3] {
        self.velocity
    }

    /// What interval a step will accept.
    #[must_use]
    pub const fn interval(&self) -> IntervalBand {
        self.interval
    }

    /// Replaces the accepted interval band.
    pub const fn set_interval(&mut self, interval: IntervalBand) {
        self.interval = interval;
    }
}

/// A planar rigid body with mass and rotational inertia.
///
/// Wrenches accumulate until the next [`RigidBody::step`], which applies
/// them and clears them, so several contacts in one frame add up rather
/// than the last one winning.
pub trait RigidBody {
    /// The shared state.
    fn state(&self) -> &BodyState;

    /// The shared state, mutably.
    fn state_mut(&mut self) -> &mut BodyState;

    /// Rotational inertia about the center of mass, kilogram meters squared.
    fn inertia(&self) -> f64;

    /// Radius of the circle enclosing the body, meters.
    fn bounding_radius(&self) -> f64;

    /// Adds a wrench to be applied at the next step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a component is not a real number.
    /// A NaN force would reach the pose through the integrator and stay
    /// there, and every later comparison against that pose would be false.
    fn apply_wrench(&mut self, fx: f64, fy: f64, torque: f64) -> Result<(), Error> {
        for (quantity, value) in [("force x", fx), ("force y", fy), ("torque", torque)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let state = self.state_mut();
        state.force[0] += fx;
        state.force[1] += fy;
        state.torque += torque;
        Ok(())
    }

    /// Integrates forward by `dt` seconds and clears the accumulated wrench.
    ///
    /// Explicit Euler, which is what the Python did. It is the cheapest
    /// integrator and the least accurate, and at the contact stiffnesses
    /// the actuator array uses it is the step size rather than the order
    /// that decides whether the result is sensible.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] or [`Error::OutOfRange`] when `dt`
    /// leaves the state's interval band, per `FR-INV-10`.
    fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.state().interval.check(dt)?;
        let inertia = self.inertia();
        if !(inertia.is_finite() && inertia > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "inertia",
                value: inertia,
                bound: "(0, inf)",
            });
        }

        let state = self.state_mut();
        let acceleration = [state.force[0] / state.mass, state.force[1] / state.mass];
        let angular = state.torque / inertia;

        state.velocity[0] = acceleration[0].mul_add(dt, state.velocity[0]);
        state.velocity[1] = acceleration[1].mul_add(dt, state.velocity[1]);
        state.velocity[2] = angular.mul_add(dt, state.velocity[2]);

        state.pose[0] = state.velocity[0].mul_add(dt, state.pose[0]);
        state.pose[1] = state.velocity[1].mul_add(dt, state.pose[1]);
        state.pose[2] = state.velocity[2].mul_add(dt, state.pose[2]);

        state.force = [0.0; 2];
        state.torque = 0.0;
        Ok(())
    }

    /// Returns the body to `(x, y, heading)` at rest, wrench cleared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a component is not a real number.
    fn reset(&mut self, x: f64, y: f64, heading: f64) -> Result<(), Error> {
        for (quantity, value) in [("x", x), ("y", y), ("heading", heading)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let state = self.state_mut();
        state.pose = [x, y, heading];
        state.velocity = [0.0; 3];
        state.force = [0.0; 2];
        state.torque = 0.0;
        Ok(())
    }
}

/// A uniform disk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleBody {
    state: BodyState,
    radius: f64,
}

impl CircleBody {
    /// Builds a disk of `mass` and `radius` at `(x, y)` facing `heading`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the mass or the radius is not
    /// finite and strictly positive, and otherwise as [`BodyState::new`].
    pub fn new(mass: f64, radius: f64, x: f64, y: f64, heading: f64) -> Result<Self, Error> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "radius",
                value: radius,
                bound: "(0, inf)",
            });
        }
        Ok(Self {
            state: BodyState::new(mass, x, y, heading)?,
            radius,
        })
    }

    /// The radius, meters.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }
}

impl RigidBody for CircleBody {
    fn state(&self) -> &BodyState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BodyState {
        &mut self.state
    }

    /// `m r^2 / 2`, a uniform disk about its center.
    fn inertia(&self) -> f64 {
        self.state.mass * self.radius * self.radius / 2.0
    }

    fn bounding_radius(&self) -> f64 {
        self.radius
    }
}

/// A uniform square plate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareBody {
    state: BodyState,
    side_length: f64,
}

impl SquareBody {
    /// Builds a square of `mass` and `side_length` at `(x, y)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the mass or the side is not
    /// finite and strictly positive, and otherwise as [`BodyState::new`].
    pub fn new(mass: f64, side_length: f64, x: f64, y: f64, heading: f64) -> Result<Self, Error> {
        if !(side_length.is_finite() && side_length > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "side length",
                value: side_length,
                bound: "(0, inf)",
            });
        }
        Ok(Self {
            state: BodyState::new(mass, x, y, heading)?,
            side_length,
        })
    }

    /// The side length, meters.
    #[must_use]
    pub const fn side_length(&self) -> f64 {
        self.side_length
    }

    /// The four corners in world frame, counterclockwise from the near left.
    #[must_use]
    pub fn corners(&self) -> [(f64, f64); 4] {
        let half = self.side_length / 2.0;
        let pose = self.state.pose;
        let (sine, cosine) = pose[2].sin_cos();
        let body = [(-half, -half), (half, -half), (half, half), (-half, half)];
        body.map(|(x, y)| {
            (
                cosine.mul_add(x, -(sine * y)) + pose[0],
                sine.mul_add(x, cosine * y) + pose[1],
            )
        })
    }
}

impl RigidBody for SquareBody {
    fn state(&self) -> &BodyState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BodyState {
        &mut self.state
    }

    /// `m a^2 / 6`, a uniform square plate about its center.
    fn inertia(&self) -> f64 {
        self.state.mass * self.side_length * self.side_length / 6.0
    }

    /// Half the diagonal, which is what encloses a square.
    fn bounding_radius(&self) -> f64 {
        self.side_length * core::f64::consts::SQRT_2 / 2.0
    }
}
