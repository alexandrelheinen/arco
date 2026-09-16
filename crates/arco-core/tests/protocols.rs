// Copyright 2026 alexandre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The protocol traits are implementable, and their contracts hold.
//!
//! These were `runtime_checkable` Python protocols, which verify a method
//! name and nothing about its signature. The point of the port is that a
//! wrong signature now fails to compile, so this file exists to prove the
//! shapes are usable and to show what an implementation looks like.

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{
    AvoidanceStrategy, Command, NearestObstacle, Occupancy, Sampler, SegmentChecker, Steerer,
    TelemetryPublisher, VehicleModel,
};
use arco_core::rng::Pcg64;

/// A box-shaped free space with one obstacle at the origin.
struct SingleObstacle {
    dimension: usize,
    radius: f64,
}

impl Occupancy for SingleObstacle {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error> {
        arco_core::geometry::require_dimension("point", point, self.dimension)?;
        arco_core::geometry::require_finite("point", point)?;
        let origin = vec![0.0; self.dimension];
        let distance = arco_core::geometry::euclidean_distance(point, &origin)?;
        Ok(NearestObstacle {
            distance: (distance - self.radius).max(0.0),
            point: origin,
        })
    }

    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error> {
        Ok(self.nearest_obstacle(point)?.distance <= 0.0)
    }
}

/// Samples uniformly inside a hypercube.
struct BoxSampler {
    bounds: Vec<(f64, f64)>,
}

impl Sampler for BoxSampler {
    fn sample(&self, generator: &mut Pcg64) -> Result<Vec<f64>, Error> {
        if self.bounds.is_empty() {
            return Err(Error::TooFew {
                quantity: "bounds",
                minimum: 1,
                actual: 0,
            });
        }
        Ok(self
            .bounds
            .iter()
            .map(|&(low, high)| low + generator.next_f64() * (high - low))
            .collect())
    }
}

/// Steers along the straight line, capped at one step.
struct LineSteerer {
    step_size: f64,
}

impl Steerer for LineSteerer {
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error> {
        arco_core::geometry::require_dimension("target", to, from.len())?;
        let distance = arco_core::geometry::euclidean_distance(from, to)?;
        if distance <= self.step_size || distance == 0.0 {
            return Ok(to.to_vec());
        }
        let scale = self.step_size / distance;
        Ok(from
            .iter()
            .zip(to)
            .map(|(start, end)| start + (end - start) * scale)
            .collect())
    }
}

/// Checks a segment by sampling it at a fixed count.
struct SampledSegments<'occupancy> {
    occupancy: &'occupancy SingleObstacle,
    sample_count: usize,
}

impl SegmentChecker for SampledSegments<'_> {
    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        arco_core::geometry::require_dimension("target", to, from.len())?;
        for index in 0..=self.sample_count {
            let ratio = f64::from(u32::try_from(index).unwrap_or_default())
                / f64::from(u32::try_from(self.sample_count).unwrap_or(1));
            let point: Vec<f64> = from
                .iter()
                .zip(to)
                .map(|(start, end)| start + (end - start) * ratio)
                .collect();
            if self.occupancy.is_occupied(&point)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// A unicycle advanced by explicit Euler.
struct Unicycle {
    pose: Pose,
    speed: f64,
    turn_rate: f64,
}

impl VehicleModel for Unicycle {
    fn pose(&self) -> Pose {
        self.pose
    }

    fn speed(&self) -> f64 {
        self.speed
    }

    fn turn_rate(&self) -> f64 {
        self.turn_rate
    }

    fn step(&mut self, command: Command, dt: f64) -> Result<(), Error> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "dt",
                value: dt,
                bound: "(0, inf)",
            });
        }
        for (quantity, value) in [("speed", command.speed), ("turn_rate", command.turn_rate)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let heading = self.pose.heading();
        self.pose = Pose::new(
            self.pose.x() + command.speed * heading.cos() * dt,
            self.pose.y() + command.speed * heading.sin() * dt,
            heading + command.turn_rate * dt,
        )?;
        self.speed = command.speed;
        self.turn_rate = command.turn_rate;
        Ok(())
    }
}

/// Pushes away from the origin.
struct PushFromOrigin;

impl AvoidanceStrategy for PushFromOrigin {
    fn turn_rate_bias(&self, pose: Pose) -> Result<f64, Error> {
        let distance = pose.x().hypot(pose.y());
        if distance == 0.0 {
            return Ok(0.0);
        }
        Ok(1.0 / distance)
    }
}

/// Counts what it is given.
#[derive(Default)]
struct CountingSink {
    count: usize,
}

impl TelemetryPublisher for CountingSink {
    type Snapshot = usize;

    fn publish(&mut self, _snapshot: &usize) {
        self.count = self.count.saturating_add(1);
    }
}

#[test]
fn an_occupancy_rejects_a_point_of_the_wrong_dimension() {
    // FR-CORE-03.
    let occupancy = SingleObstacle {
        dimension: 2,
        radius: 1.0,
    };
    let error = occupancy.is_occupied(&[0.0, 0.0, 0.0]).unwrap_err();
    assert!(
        matches!(error, Error::DimensionMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn an_occupancy_rejects_a_non_finite_query() {
    // FR-SAFE-07.
    let occupancy = SingleObstacle {
        dimension: 2,
        radius: 1.0,
    };
    let error = occupancy.is_occupied(&[f64::NAN, 0.0]).unwrap_err();
    assert!(matches!(error, Error::NotFinite { .. }), "{error:?}");
}

#[test]
fn a_point_inside_the_obstacle_is_occupied() {
    let occupancy = SingleObstacle {
        dimension: 2,
        radius: 1.0,
    };
    assert!(occupancy.is_occupied(&[0.5, 0.0]).unwrap());
    assert!(!occupancy.is_occupied(&[5.0, 0.0]).unwrap());
}

#[test]
fn a_sampler_draws_inside_its_bounds_and_repeats_under_a_seed() {
    let sampler = BoxSampler {
        bounds: vec![(-1.0, 1.0), (10.0, 20.0)],
    };
    let draw = |seed| {
        let mut generator = Pcg64::seed_from_u64(seed);
        (0..64)
            .map(|_| sampler.sample(&mut generator).unwrap())
            .collect::<Vec<_>>()
    };

    let first = draw(11);
    for state in &first {
        assert!((-1.0..1.0).contains(&state[0]), "{state:?}");
        assert!((10.0..20.0).contains(&state[1]), "{state:?}");
    }
    assert_eq!(first, draw(11), "FR-RNG-01: a seed replays identically");
}

#[test]
fn an_empty_bound_set_is_rejected_rather_than_sampled() {
    let sampler = BoxSampler { bounds: Vec::new() };
    let mut generator = Pcg64::seed_from_u64(0);
    let error = sampler.sample(&mut generator).unwrap_err();
    assert!(matches!(error, Error::TooFew { .. }), "{error:?}");
}

#[test]
fn steering_never_overshoots_its_step_size() {
    let steerer = LineSteerer { step_size: 0.5 };
    let from = [0.0, 0.0];
    let to = [10.0, 0.0];
    let stepped = steerer.steer(&from, &to).unwrap();
    let travelled = arco_core::geometry::euclidean_distance(&from, &stepped).unwrap();
    assert!(travelled <= 0.5 + 1e-12, "{travelled}");
}

#[test]
fn steering_reaches_a_target_already_within_one_step() {
    let steerer = LineSteerer { step_size: 5.0 };
    let reached = steerer.steer(&[0.0, 0.0], &[1.0, 0.0]).unwrap();
    assert_eq!(reached, vec![1.0, 0.0]);
}

#[test]
fn a_segment_through_the_obstacle_is_not_free() {
    let occupancy = SingleObstacle {
        dimension: 2,
        radius: 1.0,
    };
    let checker = SampledSegments {
        occupancy: &occupancy,
        sample_count: 32,
    };
    assert!(!checker.is_segment_free(&[-5.0, 0.0], &[5.0, 0.0]).unwrap());
    assert!(checker.is_segment_free(&[-5.0, 5.0], &[5.0, 5.0]).unwrap());
}

#[test]
fn a_vehicle_rejects_a_non_positive_interval() {
    // FR-INV-10.
    let mut vehicle = Unicycle {
        pose: Pose::new(0.0, 0.0, 0.0).unwrap(),
        speed: 0.0,
        turn_rate: 0.0,
    };
    let command = Command {
        speed: 1.0,
        turn_rate: 0.0,
    };
    for interval in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            vehicle.step(command, interval).is_err(),
            "accepted dt of {interval}"
        );
    }
}

#[test]
fn a_vehicle_rejects_a_non_finite_command() {
    let mut vehicle = Unicycle {
        pose: Pose::new(0.0, 0.0, 0.0).unwrap(),
        speed: 0.0,
        turn_rate: 0.0,
    };
    let command = Command {
        speed: f64::NAN,
        turn_rate: 0.0,
    };
    assert!(vehicle.step(command, 0.1).is_err());
}

#[test]
fn a_vehicle_driven_straight_advances_along_its_heading() {
    let mut vehicle = Unicycle {
        pose: Pose::new(0.0, 0.0, 0.0).unwrap(),
        speed: 0.0,
        turn_rate: 0.0,
    };
    let command = Command {
        speed: 2.0,
        turn_rate: 0.0,
    };
    vehicle.step(command, 0.5).unwrap();
    assert!((vehicle.pose().x() - 1.0).abs() < 1e-12);
    assert!(vehicle.pose().y().abs() < 1e-12);
    assert!((vehicle.speed() - 2.0).abs() < 1e-12);
    assert!(vehicle.turn_rate().abs() < 1e-12);
}

#[test]
fn a_vehicle_keeps_its_heading_wrapped_while_turning() {
    // FR-INV-15: a heading that accumulates past a full turn stays wrapped.
    let mut vehicle = Unicycle {
        pose: Pose::new(0.0, 0.0, 0.0).unwrap(),
        speed: 0.0,
        turn_rate: 0.0,
    };
    let command = Command {
        speed: 0.0,
        turn_rate: 1.0,
    };
    for _ in 0..100 {
        vehicle.step(command, 0.1).unwrap();
        let heading = vehicle.pose().heading();
        assert!(
            (-core::f64::consts::PI..core::f64::consts::PI).contains(&heading),
            "{heading}"
        );
    }
}

#[test]
fn an_avoidance_bias_grows_as_the_obstacle_nears() {
    let strategy = PushFromOrigin;
    let near = strategy
        .turn_rate_bias(Pose::new(1.0, 0.0, 0.0).unwrap())
        .unwrap();
    let far = strategy
        .turn_rate_bias(Pose::new(10.0, 0.0, 0.0).unwrap())
        .unwrap();
    assert!(near > far, "{near} against {far}");
}

#[test]
fn a_telemetry_sink_receives_every_snapshot() {
    let mut sink = CountingSink::default();
    for index in 0..5_usize {
        sink.publish(&index);
    }
    assert_eq!(sink.count, 5);
}
