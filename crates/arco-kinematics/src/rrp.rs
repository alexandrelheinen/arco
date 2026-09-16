//! Two-link planar arm on a prismatic lift.

use arco_core::Error;

use crate::rr::{JointAngles, RrRobot};

/// The configuration of an RRP arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Configuration {
    /// The two revolute joints, radians.
    pub angles: JointAngles,
    /// The prismatic joint height, meters.
    pub height: f64,
}

/// A planar two-link arm mounted on a vertical prismatic joint.
///
/// The revolute pair behaves exactly as [`RrRobot`], so this owns one
/// rather than repeating its trigonometry, and adds a bounded lift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrpRobot {
    arm: RrRobot,
    minimum_height: f64,
    maximum_height: f64,
}

impl RrpRobot {
    /// Builds an arm from its link lengths and its lift range, meters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when a link length is not strictly
    /// positive, when a height bound is not finite, or when the range is
    /// empty.
    pub fn new(
        first_link_length: f64,
        second_link_length: f64,
        minimum_height: f64,
        maximum_height: f64,
    ) -> Result<Self, Error> {
        let arm = RrRobot::new(first_link_length, second_link_length)?;
        for (quantity, value) in [
            ("minimum_height", minimum_height),
            ("maximum_height", maximum_height),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        if maximum_height <= minimum_height {
            return Err(Error::OutOfRange {
                quantity: "maximum_height",
                value: maximum_height,
                bound: "(minimum_height, inf)",
            });
        }
        Ok(Self {
            arm,
            minimum_height,
            maximum_height,
        })
    }

    /// The planar arm this one is mounted on.
    #[must_use]
    pub const fn arm(self) -> RrRobot {
        self.arm
    }

    /// The lift range as `(minimum, maximum)`, meters.
    #[must_use]
    pub const fn height_range(self) -> (f64, f64) {
        (self.minimum_height, self.maximum_height)
    }

    /// The reachable annulus in the plane, meters.
    #[must_use]
    pub fn workspace_annulus(self) -> (f64, f64) {
        self.arm.workspace_annulus()
    }

    /// The end effector position for a configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the height leaves the lift
    /// range, per `FR-INV-14`, and otherwise as
    /// [`RrRobot::forward_kinematics`].
    pub fn forward_kinematics(
        self,
        configuration: Configuration,
    ) -> Result<(f64, f64, f64), Error> {
        self.require_reachable_height(configuration.height)?;
        let (x, y) = self.arm.forward_kinematics(configuration.angles)?;
        Ok((x, y, configuration.height))
    }

    /// Joint angles placing the end effector over a target, if reachable.
    ///
    /// The lift is independent of the revolute pair, so the planar
    /// solution is the arm's and the height is whatever the caller asks
    /// for, checked against the range.
    ///
    /// # Errors
    ///
    /// As [`RrRobot::inverse_kinematics`], plus [`Error::OutOfRange`] when
    /// `height` leaves the lift range.
    pub fn inverse_kinematics(
        self,
        x: f64,
        y: f64,
        height: f64,
        tolerance: f64,
    ) -> Result<Vec<Configuration>, Error> {
        self.require_reachable_height(height)?;
        Ok(self
            .arm
            .inverse_kinematics(x, y, tolerance)?
            .into_iter()
            .map(|angles| Configuration { angles, height })
            .collect())
    }

    /// The base, shoulder, elbow and end effector positions, meters.
    ///
    /// # Errors
    ///
    /// As [`RrpRobot::forward_kinematics`].
    pub fn link_segments(
        self,
        configuration: Configuration,
    ) -> Result<[(f64, f64, f64); 4], Error> {
        self.require_reachable_height(configuration.height)?;
        let planar = self.arm.link_segments(configuration.angles)?;
        let lift = configuration.height;
        Ok([
            (0.0, 0.0, self.minimum_height),
            (planar[0].0, planar[0].1, lift),
            (planar[1].0, planar[1].1, lift),
            (planar[2].0, planar[2].1, lift),
        ])
    }

    fn require_reachable_height(self, height: f64) -> Result<(), Error> {
        if !height.is_finite() {
            return Err(Error::NotFinite {
                quantity: "height",
                value: height,
            });
        }
        if height < self.minimum_height || height > self.maximum_height {
            return Err(Error::OutOfRange {
                quantity: "height",
                value: height,
                bound: "[minimum_height, maximum_height]",
            });
        }
        Ok(())
    }
}
