//! Two-link planar revolute-revolute arm.

use arco_core::Error;
use arco_core::numeric::wrap_angle;

/// Joint angles placing the end effector somewhere, radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointAngles {
    /// First joint, measured from the positive first axis.
    pub shoulder: f64,
    /// Second joint, measured relative to the first link.
    pub elbow: f64,
}

/// A two-link planar arm with revolute joints.
///
/// Both links are rigid and the arm moves in a plane, so the reachable
/// set is the annulus between `|l1 - l2|` and `l1 + l2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrRobot {
    first_link_length: f64,
    second_link_length: f64,
}

impl RrRobot {
    /// Builds an arm from its two link lengths, meters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when either length is not strictly
    /// positive and finite.
    pub fn new(first_link_length: f64, second_link_length: f64) -> Result<Self, Error> {
        for (quantity, value) in [
            ("first_link_length", first_link_length),
            ("second_link_length", second_link_length),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }
        Ok(Self {
            first_link_length,
            second_link_length,
        })
    }

    /// Length of the first link, meters.
    #[must_use]
    pub const fn first_link_length(self) -> f64 {
        self.first_link_length
    }

    /// Length of the second link, meters.
    #[must_use]
    pub const fn second_link_length(self) -> f64 {
        self.second_link_length
    }

    /// The reachable annulus as `(inner, outer)` radii, meters.
    #[must_use]
    pub fn workspace_annulus(self) -> (f64, f64) {
        (
            (self.first_link_length - self.second_link_length).abs(),
            self.first_link_length + self.second_link_length,
        )
    }

    /// The outer reach, meters.
    #[must_use]
    pub fn workspace_radius(self) -> f64 {
        self.workspace_annulus().1
    }

    /// The end effector position for a set of joint angles.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when an angle is NaN or infinite.
    pub fn forward_kinematics(self, angles: JointAngles) -> Result<(f64, f64), Error> {
        let shoulder = wrap_angle(angles.shoulder)?;
        let elbow = wrap_angle(angles.elbow)?;
        let combined = shoulder + elbow;
        Ok((
            self.first_link_length
                .mul_add(shoulder.cos(), self.second_link_length * combined.cos()),
            self.first_link_length
                .mul_add(shoulder.sin(), self.second_link_length * combined.sin()),
        ))
    }

    /// The shoulder, elbow and end effector positions, meters.
    ///
    /// # Errors
    ///
    /// As [`RrRobot::forward_kinematics`].
    pub fn link_segments(self, angles: JointAngles) -> Result<[(f64, f64); 3], Error> {
        let shoulder = wrap_angle(angles.shoulder)?;
        let elbow_position = (
            self.first_link_length * shoulder.cos(),
            self.first_link_length * shoulder.sin(),
        );
        Ok([(0.0, 0.0), elbow_position, self.forward_kinematics(angles)?])
    }

    /// Joint angles placing the end effector at a target, if reachable.
    ///
    /// Returns both branches, elbow down first and elbow up second, and an
    /// empty vector for a target outside the annulus. On the annulus
    /// boundary the two branches coincide.
    ///
    /// `FR-INV-14`: a returned solution is always inside the reachable set
    /// and always reproduces the target under forward kinematics, and a
    /// target outside the set is refused rather than approximated.
    ///
    /// # Arguments
    ///
    /// * `x` - Target along the first axis, meters.
    /// * `y` - Target along the second axis, meters.
    /// * `tolerance` - Slack on the annulus boundary, meters, absorbing
    ///   the rounding a caller's own arithmetic introduced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a coordinate is NaN or infinite,
    /// or [`Error::OutOfRange`] when `tolerance` is negative.
    pub fn inverse_kinematics(
        self,
        x: f64,
        y: f64,
        tolerance: f64,
    ) -> Result<Vec<JointAngles>, Error> {
        for (quantity, value) in [("x", x), ("y", y)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(Error::OutOfRange {
                quantity: "tolerance",
                value: tolerance,
                bound: "[0, inf)",
            });
        }

        let radius_squared = x.mul_add(x, y * y);
        let radius = radius_squared.sqrt();
        let (inner, outer) = self.workspace_annulus();
        if radius > outer + tolerance || radius < inner - tolerance {
            return Ok(Vec::new());
        }

        let denominator = 2.0 * self.first_link_length * self.second_link_length;
        let cosine = (radius_squared
            - self.first_link_length.mul_add(
                self.first_link_length,
                self.second_link_length * self.second_link_length,
            ))
            / denominator;
        // Clamped because a target on the boundary lands just outside the
        // domain of acos through ordinary rounding, and acos of 1.0000001
        // is NaN rather than zero.
        let cosine = cosine.clamp(-1.0, 1.0);

        let mut solutions = Vec::with_capacity(2);
        for sign in [1.0_f64, -1.0] {
            let elbow = sign * cosine.acos();
            let along = self
                .second_link_length
                .mul_add(elbow.cos(), self.first_link_length);
            let across = self.second_link_length * elbow.sin();
            let shoulder = y.atan2(x) - across.atan2(along);
            solutions.push(JointAngles {
                shoulder: wrap_angle(shoulder)?,
                elbow: wrap_angle(elbow)?,
            });
        }
        Ok(solutions)
    }
}
