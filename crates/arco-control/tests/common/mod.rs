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

//! Fixtures the control tests share.
//!
//! The vehicle here is a unicycle, which is what the tracking loop was
//! written against. The real one lands with `arco-guidance` in phase 8;
//! this exists so the loop can be tested before it does.

// A shared test module is compiled into every test binary that declares
// it, so an item only one of them uses looks dead in the others, and an
// item nothing re-exports looks unreachable.
#![expect(
    dead_code,
    unreachable_pub,
    reason = "each test binary uses a subset of these fixtures"
)]
// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{AvoidanceStrategy, Command, VehicleModel};

/// A unicycle: it goes where it points, at the speed it was told.
#[derive(Debug, Clone, Copy)]
pub struct Unicycle {
    pose: Pose,
    speed: f64,
    turn_rate: f64,
}

impl Unicycle {
    /// Builds a unicycle at rest.
    ///
    /// # Panics
    ///
    /// Panics when the pose is not a real number, which in a fixture is
    /// the test being wrong rather than the code.
    #[must_use]
    pub fn at(x: f64, y: f64, heading: f64) -> Self {
        Self {
            pose: Pose::new(x, y, heading).expect("fixture pose is finite"),
            speed: 0.0,
            turn_rate: 0.0,
        }
    }
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
                quantity: "elapsed interval",
                value: dt,
                bound: "(0, inf)",
            });
        }
        for (quantity, value) in [("speed", command.speed), ("turn rate", command.turn_rate)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let heading = command.turn_rate.mul_add(dt, self.pose.heading());
        let (sine, cosine) = self.pose.heading().sin_cos();
        self.pose = Pose::new(
            (command.speed * cosine).mul_add(dt, self.pose.x()),
            (command.speed * sine).mul_add(dt, self.pose.y()),
            heading,
        )?;
        self.speed = command.speed;
        self.turn_rate = command.turn_rate;
        Ok(())
    }
}

/// Avoidance that never biases anything.
#[derive(Debug, Clone, Copy)]
pub struct NoAvoidance;

impl AvoidanceStrategy for NoAvoidance {
    fn turn_rate_bias(&self, _pose: Pose) -> Result<f64, Error> {
        Ok(0.0)
    }
}

/// Avoidance that always biases by a fixed amount.
#[derive(Debug, Clone, Copy)]
pub struct FixedBias(pub f64);

impl AvoidanceStrategy for FixedBias {
    fn turn_rate_bias(&self, _pose: Pose) -> Result<f64, Error> {
        Ok(self.0)
    }
}

/// A straight path along the first axis, one meter apart.
#[must_use]
pub fn straight_path(count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|index| (f64::from(u32::try_from(index).unwrap_or(u32::MAX)), 0.0))
        .collect()
}
