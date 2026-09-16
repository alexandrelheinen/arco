//! An array of contact actuators arranged around a rigid body.
//!
//! Each actuator sits at an angle around the body, a standoff beyond its
//! bounding radius, and pushes inward. A grasp matrix maps the per
//! actuator forces to a wrench on the body; allocation inverts it. Both
//! axes of every actuator, angular and radial, are second-order closed
//! loops driven toward a setpoint, so the array is a plant as well as a
//! controller and a caller has to integrate it.

use arco_core::Error;
use arco_core::protocols::{NearestObstacle, Occupancy};

use crate::body::RigidBody;
use crate::limits::IntervalBand;
use crate::linalg::Symmetric3;

/// The fewest actuators that can produce a wrench in every direction.
///
/// Two can only push along one line between them, so the grasp matrix
/// cannot reach a general planar wrench and the allocation below would
/// quietly return the least-squares miss instead of the answer.
const MINIMUM_ACTUATORS: usize = 3;

/// A caller-supplied nearest-hazard query.
type HazardQuery = Box<dyn Fn(&[f64]) -> Result<NearestObstacle, Error> + Send + Sync>;

/// A hazard and how far the querying actuator is from it, meters.
type Hazard = (f64, (f64, f64));

/// Where the nearest hazard is, for the repulsive field.
///
/// ADR-004. The field queries this once per actuator per step, so a
/// caller-supplied closure on that path costs what the crossing costs.
/// The native variants never leave Rust.
pub enum HazardPolicy<O> {
    /// Nothing to avoid.
    None,
    /// Query an occupancy map.
    Occupancy(O),
    /// Anything the caller supplied.
    ///
    /// Replaces the `nearest_obstacle_fn` argument. Slower, per deviation
    /// A-07.
    Custom(HazardQuery),
}

impl<O> core::fmt::Debug for HazardPolicy<O> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match *self {
            Self::None => "None",
            Self::Occupancy(_) => "Occupancy",
            Self::Custom(_) => "Custom",
        };
        formatter.write_str(name)
    }
}

impl<O: Occupancy> HazardPolicy<O> {
    /// The nearest hazard to `point`, measured from its center.
    ///
    /// From the center rather than from the surface, because the
    /// repulsive field's influence radius is expressed that way and
    /// because that is what Python's `nearest_obstacle_fn` returned. See
    /// deviation A-16.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy or the custom query returns.
    pub fn nearest(&self, point: &[f64]) -> Result<Option<NearestObstacle>, Error> {
        match self {
            Self::None => Ok(None),
            Self::Occupancy(occupancy) => {
                let mut nearest = occupancy.nearest_obstacle(point)?;
                nearest.distance += occupancy.clearance();
                Ok(Some(nearest))
            }
            Self::Custom(query) => Ok(Some(query(point)?)),
        }
    }
}

/// How the actuator loops behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActuatorSettings {
    /// How far beyond the bounding radius an actuator rests, meters.
    pub standoff: f64,
    /// Natural frequency of both actuator loops, radians per second.
    pub natural_frequency: f64,
    /// Damping ratio of both actuator loops.
    pub damping_ratio: f64,
    /// Contact spring stiffness, newtons per meter.
    pub spring_stiffness: f64,
    /// What elapsed interval [`ActuatorArray::step`] will accept.
    pub interval: IntervalBand,
}

impl Default for ActuatorSettings {
    /// What `arco.control.actuator.ActuatorArray` defaulted to.
    fn default() -> Self {
        Self {
            standoff: 0.05,
            natural_frequency: 10.0,
            damping_ratio: 0.7,
            spring_stiffness: 100.0,
            interval: IntervalBand::default(),
        }
    }
}

/// One axis of the actuator array: position, velocity and setpoint.
#[derive(Debug, Clone, PartialEq)]
struct Axis {
    position: Vec<f64>,
    velocity: Vec<f64>,
    reference: Vec<f64>,
}

impl Axis {
    /// Builds an axis resting at `position` with that as its setpoint.
    fn at_rest(position: Vec<f64>) -> Self {
        Self {
            velocity: vec![0.0; position.len()],
            reference: position.clone(),
            position,
        }
    }

    /// Integrates one step of `x'' = -a x' - b (x - x*)`.
    ///
    /// Forward Euler, position first. The ordering is not incidental: the
    /// velocity-first arrangement gives a discrete state matrix whose
    /// spectral radius exceeds one at the step sizes this array is used
    /// with, so the actuators diverge rather than settle.
    fn integrate(&mut self, damping: f64, stiffness: f64, dt: f64) {
        for index in 0..self.position.len() {
            let position = self.position.get(index).copied().unwrap_or_default();
            let velocity = self.velocity.get(index).copied().unwrap_or_default();
            let reference = self.reference.get(index).copied().unwrap_or_default();
            let acceleration = stiffness.mul_add(-(position - reference), -(damping * velocity));
            if let Some(slot) = self.position.get_mut(index) {
                *slot = velocity.mul_add(dt, position);
            }
            if let Some(slot) = self.velocity.get_mut(index) {
                *slot = acceleration.mul_add(dt, velocity);
            }
        }
    }
}

/// The three by two-N map from actuator forces to a body wrench.
///
/// Column `2i` is actuator `i` pushing inward, column `2i + 1` is the same
/// actuator pushing along the tangent, counterclockwise positive.
#[derive(Debug, Clone, PartialEq)]
pub struct GraspMatrix {
    columns: Vec<[f64; 3]>,
}

impl GraspMatrix {
    /// How many columns, which is twice the actuator count.
    #[must_use]
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// The wrench produced by a force vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `forces` is not as wide
    /// as the matrix.
    pub fn wrench(&self, forces: &[f64]) -> Result<[f64; 3], Error> {
        if forces.len() != self.columns.len() {
            return Err(Error::DimensionMismatch {
                quantity: "actuator forces",
                expected: self.columns.len(),
                actual: forces.len(),
            });
        }
        let mut total = [0.0_f64; 3];
        for (column, force) in self.columns.iter().zip(forces) {
            for (slot, entry) in total.iter_mut().zip(column) {
                *slot += entry * force;
            }
        }
        Ok(total)
    }

    /// Allocates a force vector producing `wrench` as closely as possible.
    ///
    /// The minimum-norm least-squares solution, which is the Moore-Penrose
    /// pseudo-inverse applied to the wrench. Where the wrench is reachable
    /// the result produces it exactly; where it is not, the result is the
    /// closest the array can come, and nothing here reports which of those
    /// happened. A caller that needs to know multiplies the answer back
    /// through [`GraspMatrix::wrench`] and compares.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the matrix or the wrench carries
    /// a value that is not a real number.
    pub fn allocate(&self, wrench: [f64; 3], stride: usize) -> Result<Vec<f64>, Error> {
        let stride = stride.max(1);
        let mut normal = Symmetric3::zero();
        for column in self.columns.iter().step_by(stride) {
            for row in 0..3 {
                for other in row..3 {
                    let left = column.get(row).copied().unwrap_or_default();
                    let right = column.get(other).copied().unwrap_or_default();
                    normal.add(row, other, left * right);
                }
            }
        }

        // `A+ w = A^T (A A^T)+ w`, so solving the three by three normal
        // system first turns a two-N by three multiplication into a
        // three-vector one.
        let scaled = normal.solve_pseudo(wrench)?;
        let mut forces = vec![0.0; self.columns.len()];
        for (index, column) in self.columns.iter().enumerate().step_by(stride) {
            let mut value = 0.0;
            for (entry, factor) in column.iter().zip(&scaled) {
                value += entry * factor;
            }
            if let Some(slot) = forces.get_mut(index) {
                *slot = value;
            }
        }
        Ok(forces)
    }
}

/// N contact actuators arranged around a planar rigid body.
#[derive(Debug, Clone)]
pub struct ActuatorArray {
    settings: ActuatorSettings,
    angular: Axis,
    radial: Option<Axis>,
}

impl ActuatorArray {
    /// Builds an array of `actuator_count` actuators, evenly spaced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when fewer than three are asked for, and
    /// [`Error::OutOfRange`] when a setting is not finite, the natural
    /// frequency or the stiffness is not positive, the damping ratio is
    /// negative, or the standoff is negative.
    pub fn new(actuator_count: usize, settings: ActuatorSettings) -> Result<Self, Error> {
        if actuator_count < MINIMUM_ACTUATORS {
            return Err(Error::TooFew {
                quantity: "actuators",
                minimum: MINIMUM_ACTUATORS,
                actual: actuator_count,
            });
        }
        for (quantity, value) in [
            ("natural frequency", settings.natural_frequency),
            ("spring stiffness", settings.spring_stiffness),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }
        for (quantity, value) in [
            ("damping ratio", settings.damping_ratio),
            ("standoff", settings.standoff),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "[0, inf)",
                });
            }
        }

        let spacing = core::f64::consts::TAU / count_as_f64(actuator_count);
        let angles = (0..actuator_count)
            .map(|index| count_as_f64(index) * spacing)
            .collect();
        Ok(Self {
            settings,
            angular: Axis::at_rest(angles),
            radial: None,
        })
    }

    /// How many actuators the array holds.
    #[must_use]
    pub fn actuator_count(&self) -> usize {
        self.angular.position.len()
    }

    /// The settings in force.
    #[must_use]
    pub const fn settings(&self) -> ActuatorSettings {
        self.settings
    }

    /// Current placement angles, radians, in the body frame.
    #[must_use]
    pub fn angles(&self) -> &[f64] {
        &self.angular.position
    }

    /// Angular setpoints, radians.
    #[must_use]
    pub fn reference_angles(&self) -> &[f64] {
        &self.angular.reference
    }

    /// Angular rates, radians per second.
    #[must_use]
    pub fn angle_velocities(&self) -> &[f64] {
        &self.angular.velocity
    }

    /// Radial positions, meters, once the radial axis exists.
    #[must_use]
    pub fn radii(&self) -> Option<&[f64]> {
        self.radial.as_ref().map(|axis| axis.position.as_slice())
    }

    /// Radial setpoints, meters, once the radial axis exists.
    #[must_use]
    pub fn reference_radii(&self) -> Option<&[f64]> {
        self.radial.as_ref().map(|axis| axis.reference.as_slice())
    }

    /// Overwrites the placement angles, leaving the rates alone.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the count is wrong, and
    /// [`Error::NotFinite`] when an angle is not a real number.
    pub fn set_angles(&mut self, angles: &[f64]) -> Result<(), Error> {
        if angles.len() != self.actuator_count() {
            return Err(Error::DimensionMismatch {
                quantity: "actuator angles",
                expected: self.actuator_count(),
                actual: angles.len(),
            });
        }
        for &angle in angles {
            if !angle.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "actuator angle",
                    value: angle,
                });
            }
        }
        self.angular.position.clear();
        self.angular.position.extend_from_slice(angles);
        Ok(())
    }

    /// The grasp matrix for the current placement around `body`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the body's bounding radius is
    /// not a positive real number.
    pub fn grasp_matrix<B: RigidBody>(&self, body: &B) -> Result<GraspMatrix, Error> {
        let radius = body.bounding_radius();
        if !(radius.is_finite() && radius > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "bounding radius",
                value: radius,
                bound: "(0, inf)",
            });
        }
        let heading = body.state().pose().get(2).copied().unwrap_or_default();

        let mut columns = Vec::with_capacity(self.actuator_count().saturating_mul(2));
        for &angle in &self.angular.position {
            let (sine, cosine) = (angle + heading).sin_cos();
            // Contact point relative to the body center, and the two
            // directions an actuator can push in there.
            let (arm_x, arm_y) = (radius * cosine, radius * sine);
            let (normal_x, normal_y) = (-cosine, -sine);
            let (tangent_x, tangent_y) = (-sine, cosine);
            columns.push([
                normal_x,
                normal_y,
                arm_x.mul_add(normal_y, -(arm_y * normal_x)),
            ]);
            columns.push([
                tangent_x,
                tangent_y,
                arm_x.mul_add(tangent_y, -(arm_y * tangent_x)),
            ]);
        }
        Ok(GraspMatrix { columns })
    }

    /// Forces on every axis producing `wrench` as closely as possible.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::grasp_matrix`] and [`GraspMatrix::allocate`].
    pub fn allocate_forces<B: RigidBody>(
        &self,
        wrench: [f64; 3],
        body: &B,
    ) -> Result<Vec<f64>, Error> {
        self.grasp_matrix(body)?.allocate(wrench, 1)
    }

    /// Forces on the radial axes only, the tangential ones left at zero.
    ///
    /// What the spring contact model can actually deliver: a spring at the
    /// radial axis pushes inward and nothing produces a tangential force.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::allocate_forces`].
    pub fn allocate_radial_forces<B: RigidBody>(
        &self,
        wrench: [f64; 3],
        body: &B,
    ) -> Result<Vec<f64>, Error> {
        self.grasp_matrix(body)?.allocate(wrench, 2)
    }

    /// Where every actuator sits in the world frame.
    ///
    /// Uses the actual radius per actuator once the radial axis exists,
    /// and the nominal standoff before that.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::grasp_matrix`].
    pub fn actuator_positions<B: RigidBody>(&self, body: &B) -> Result<Vec<(f64, f64)>, Error> {
        let nominal = self.nominal_radius(body)?;
        let pose = body.state().pose();
        let heading = pose.get(2).copied().unwrap_or_default();
        let center = (
            pose.first().copied().unwrap_or_default(),
            pose.get(1).copied().unwrap_or_default(),
        );

        Ok(self
            .angular
            .position
            .iter()
            .enumerate()
            .map(|(index, &angle)| {
                let radius = self
                    .radial
                    .as_ref()
                    .and_then(|axis| axis.position.get(index).copied())
                    .unwrap_or(nominal);
                let (sine, cosine) = (angle + heading).sin_cos();
                (
                    radius.mul_add(cosine, center.0),
                    radius.mul_add(sine, center.1),
                )
            })
            .collect())
    }

    /// Applies a force vector to `body` through the grasp matrix.
    ///
    /// # Errors
    ///
    /// As [`GraspMatrix::wrench`] and [`RigidBody::apply_wrench`].
    pub fn apply_to_body<B: RigidBody>(&self, forces: &[f64], body: &mut B) -> Result<(), Error> {
        let wrench = self.grasp_matrix(body)?.wrench(forces)?;
        body.apply_wrench(
            wrench.first().copied().unwrap_or_default(),
            wrench.get(1).copied().unwrap_or_default(),
            wrench.get(2).copied().unwrap_or_default(),
        )
    }

    /// Points the array so its inward direction lines up with `wrench`.
    ///
    /// Rotates the whole array rather than each actuator separately, which
    /// keeps the spacing even and so keeps the precompression bias in
    /// [`ActuatorArray::compute_reference_radii`] cancelling in the net
    /// wrench. The angles computed here are setpoints; the actuators reach
    /// them through [`ActuatorArray::step`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the wrench carries a value that
    /// is not a real number.
    pub fn aim_at<B: RigidBody>(&mut self, wrench: [f64; 3], body: &B) -> Result<(), Error> {
        for value in wrench {
            if !value.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "target wrench",
                    value,
                });
            }
        }
        let force_x = wrench.first().copied().unwrap_or_default();
        let force_y = wrench.get(1).copied().unwrap_or_default();
        let heading = body.state().pose().get(2).copied().unwrap_or_default();

        let desired_world = (-force_y).atan2(-force_x) + core::f64::consts::PI;
        let desired_body = desired_world - heading;
        let count = self.actuator_count();
        let spacing = core::f64::consts::TAU / count_as_f64(count);
        self.angular.reference.clear();
        self.angular
            .reference
            .extend((0..count).map(|index| count_as_f64(index).mul_add(spacing, desired_body)));
        Ok(())
    }

    /// Creates the radial axis at the nominal contact distance.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::grasp_matrix`].
    pub fn init_radii<B: RigidBody>(&mut self, body: &B) -> Result<(), Error> {
        let nominal = self.nominal_radius(body)?;
        self.radial = Some(Axis::at_rest(vec![nominal; self.actuator_count()]));
        Ok(())
    }

    /// Sets radial setpoints so the springs settle at `desired_forces`.
    ///
    /// Inverts the spring law: a spring compressed by `F / k` pushes with
    /// `F`, so the setpoint is the nominal radius less that compression.
    /// A spring can only push, so a symmetric precompression bias is added
    /// to make every desired force non-negative. For an evenly spaced
    /// array that bias cancels in the net wrench, which is why
    /// [`ActuatorArray::aim_at`] rotates the array as a whole.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `desired_forces` is not
    /// twice the actuator count, and [`Error::NotFinite`] when one is not
    /// a real number.
    pub fn compute_reference_radii<B: RigidBody>(
        &mut self,
        desired_forces: &[f64],
        body: &B,
    ) -> Result<(), Error> {
        let count = self.actuator_count();
        if desired_forces.len() != count.saturating_mul(2) {
            return Err(Error::DimensionMismatch {
                quantity: "desired forces",
                expected: count.saturating_mul(2),
                actual: desired_forces.len(),
            });
        }
        for &force in desired_forces {
            if !force.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "desired force",
                    value: force,
                });
            }
        }
        if self.radial.is_none() {
            self.init_radii(body)?;
        }
        let nominal = self.nominal_radius(body)?;

        let radial: Vec<f64> = (0..count)
            .map(|index| {
                desired_forces
                    .get(index.saturating_mul(2))
                    .copied()
                    .unwrap_or_default()
            })
            .collect();
        let bias = radial
            .iter()
            .fold(0.0_f64, |smallest, &force| smallest.min(force))
            .min(0.0);

        if let Some(axis) = self.radial.as_mut() {
            axis.reference.clear();
            axis.reference.extend(
                radial
                    .iter()
                    .map(|&force| nominal - (force - bias) / self.settings.spring_stiffness),
            );
        }
        Ok(())
    }

    /// Integrates both actuator axes by one step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] or [`Error::OutOfRange`] when `dt`
    /// leaves the configured interval band, per `FR-INV-10`.
    pub fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.settings.interval.check(dt)?;
        let damping = 4.0 * self.settings.damping_ratio * self.settings.natural_frequency;
        let stiffness = 2.0 * self.settings.natural_frequency * self.settings.natural_frequency;

        self.angular.integrate(damping, stiffness, dt);
        if let Some(axis) = self.radial.as_mut() {
            axis.integrate(damping, stiffness, dt);
        }
        Ok(())
    }

    /// The force each spring is currently producing.
    ///
    /// A spring only pushes, so a stretched one contributes nothing rather
    /// than pulling the body back.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::grasp_matrix`].
    pub fn spring_forces<B: RigidBody>(&mut self, body: &B) -> Result<Vec<f64>, Error> {
        if self.radial.is_none() {
            self.init_radii(body)?;
        }
        let nominal = self.nominal_radius(body)?;
        let count = self.actuator_count();
        let mut forces = vec![0.0; count.saturating_mul(2)];
        for index in 0..count {
            let radius = self
                .radial
                .as_ref()
                .and_then(|axis| axis.position.get(index).copied())
                .unwrap_or(nominal);
            let compression = (nominal - radius).max(0.0);
            if let Some(slot) = forces.get_mut(index.saturating_mul(2)) {
                *slot = self.settings.spring_stiffness * compression;
            }
        }
        Ok(forces)
    }

    /// Computes the spring forces and applies them to `body`.
    ///
    /// # Errors
    ///
    /// As [`ActuatorArray::spring_forces`] and
    /// [`ActuatorArray::apply_to_body`].
    pub fn apply_spring_forces<B: RigidBody>(&mut self, body: &mut B) -> Result<(), Error> {
        let forces = self.spring_forces(body)?;
        self.apply_to_body(&forces, body)
    }

    /// The wrench a repulsive field puts on `body`.
    ///
    /// Each actuator is pushed away from whichever hazard is nearest to
    /// it, static or peer, with magnitude `stiffness (radius - distance)^2`
    /// inside the influence radius and nothing outside it. Local by
    /// construction: it knows one hazard per actuator and cannot reason
    /// about whether pushing away leads anywhere.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the stiffness or the influence
    /// radius is not a positive real number, and otherwise whatever the
    /// hazard policy returns.
    pub fn repulsive_wrench<B: RigidBody, O: Occupancy>(
        &self,
        body: &B,
        hazards: &HazardPolicy<O>,
        peers: &[(f64, f64)],
        stiffness: f64,
        influence_radius: f64,
    ) -> Result<[f64; 3], Error> {
        for (quantity, value) in [
            ("repulsion stiffness", stiffness),
            ("influence radius", influence_radius),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }

        let pose = body.state().pose();
        let center = (
            pose.first().copied().unwrap_or_default(),
            pose.get(1).copied().unwrap_or_default(),
        );
        let mut total = [0.0_f64; 3];

        for position in self.actuator_positions(body)? {
            let Some((distance, hazard)) = nearest_hazard(position, hazards, peers)? else {
                continue;
            };
            if distance >= influence_radius {
                continue;
            }

            let (offset_x, offset_y) = (position.0 - hazard.0, position.1 - hazard.1);
            let span = offset_x.hypot(offset_y);
            // Sitting exactly on the hazard leaves no direction to push
            // in, so the array pushes outward from the body instead, which
            // is the only other direction that means anything here.
            let direction = if span <= f64::EPSILON {
                let angle = (position.1 - center.1).atan2(position.0 - center.0);
                let (sine, cosine) = angle.sin_cos();
                (cosine, sine)
            } else {
                (offset_x / span, offset_y / span)
            };

            let magnitude = stiffness * (influence_radius - distance).powi(2);
            let force = (magnitude * direction.0, magnitude * direction.1);
            let arm = (position.0 - center.0, position.1 - center.1);
            if let Some(slot) = total.first_mut() {
                *slot += force.0;
            }
            if let Some(slot) = total.get_mut(1) {
                *slot += force.1;
            }
            if let Some(slot) = total.get_mut(2) {
                *slot += arm.0.mul_add(force.1, -(arm.1 * force.0));
            }
        }
        Ok(total)
    }

    /// Where an actuator rests when nothing has compressed it.
    fn nominal_radius<B: RigidBody>(&self, body: &B) -> Result<f64, Error> {
        let radius = body.bounding_radius();
        if !(radius.is_finite() && radius > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "bounding radius",
                value: radius,
                bound: "(0, inf)",
            });
        }
        Ok(radius + self.settings.standoff)
    }
}

/// The nearest hazard to `position`, static or peer, and how far it is.
fn nearest_hazard<O: Occupancy>(
    position: (f64, f64),
    hazards: &HazardPolicy<O>,
    peers: &[(f64, f64)],
) -> Result<Option<Hazard>, Error> {
    let query = [position.0, position.1];
    let mut best = hazards.nearest(&query)?.map(|nearest| {
        (
            nearest.distance,
            (
                nearest.point.first().copied().unwrap_or_default(),
                nearest.point.get(1).copied().unwrap_or_default(),
            ),
        )
    });

    for &peer in peers {
        let distance = (peer.0 - position.0).hypot(peer.1 - position.1);
        if best.is_none_or(|(previous, _)| distance < previous) {
            best = Some((distance, peer));
        }
    }
    Ok(best)
}

/// A count as a float, saturating rather than wrapping.
fn count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}
