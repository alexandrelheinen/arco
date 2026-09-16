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

//! FR-INV-10: a control step refuses an interval it cannot compute with.
//!
//! No control step here reads a clock. The interval arrives as an
//! argument, which makes a run reproducible and a test able to state its
//! sample rate, and it also means the value can be anything the caller
//! computed: negative after a clock adjustment, zero after two reads
//! inside one tick, enormous after a stall. Each of those produces a
//! command that is arithmetically fine and physically wrong.

mod common;

use arco_control::body::{CircleBody, RigidBody};
use arco_control::limits::{CommandConditioner, CommandLimits, IntervalBand};
use arco_control::pid::{PidController, PidSettings};
use arco_control::pursuit::PurePursuitTracker;
use arco_control::tracking::{TrackingLoop, TrackingSettings};
use arco_core::Error;
use arco_core::protocols::Command;

use common::{NoAvoidance, Unicycle, straight_path};

/// The intervals every step in the crate has to refuse.
const REJECTED: [f64; 6] = [0.0, -0.1, f64::NAN, f64::INFINITY, 1e-12, 60.0];

#[test]
fn a_band_rejects_its_own_degenerate_forms() {
    assert!(matches!(
        IntervalBand::new(0.0, 1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        IntervalBand::new(1.0, f64::NAN),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        IntervalBand::new(2.0, 1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(IntervalBand::new(1e-3, 1e-1).is_ok());
    // A band may be a single value, which is what a fixed-rate loop is.
    assert!(IntervalBand::new(0.01, 0.01).is_ok());
}

#[test]
fn a_band_separates_a_non_finite_interval_from_an_out_of_range_one() {
    // Different diagnoses: one says the caller's arithmetic broke, the
    // other says the loop missed its deadline.
    let band = IntervalBand::default();
    assert!(matches!(band.check(f64::NAN), Err(Error::NotFinite { .. })));
    assert!(matches!(
        band.check(f64::INFINITY),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(band.check(0.0), Err(Error::OutOfRange { .. })));
    assert!(matches!(band.check(-1.0), Err(Error::OutOfRange { .. })));
    assert!(band.check(0.1).is_ok());
}

#[test]
fn the_conditioner_refuses_an_interval_outside_the_band() {
    let mut conditioner =
        CommandConditioner::new(CommandLimits::default()).expect("defaults are consistent");
    for dt in REJECTED {
        assert!(
            conditioner
                .apply(
                    Command {
                        speed: 1.0,
                        turn_rate: 0.0
                    },
                    dt
                )
                .is_err(),
            "the conditioner accepted an interval of {dt}"
        );
    }
}

#[test]
fn the_controller_refuses_an_interval_outside_the_band() {
    let mut controller = PidController::new(PidSettings::default()).expect("defaults are valid");
    for dt in REJECTED {
        assert!(
            controller.step(0.0, 1.0, dt).is_err(),
            "the controller accepted an interval of {dt}"
        );
    }
    assert!(controller.step(0.0, 1.0, 0.1).is_ok());
}

#[test]
fn a_body_refuses_an_interval_outside_the_band() {
    let mut body = CircleBody::new(1.0, 0.5, 0.0, 0.0, 0.0).expect("a valid disk");
    for dt in REJECTED {
        assert!(
            body.step(dt).is_err(),
            "the body accepted an interval of {dt}"
        );
    }
    assert!(body.step(0.01).is_ok());
}

#[test]
fn the_tracking_loop_refuses_an_interval_outside_the_band() {
    let path = straight_path(6);
    let mut loop_ = TrackingLoop::new(
        Unicycle::at(0.0, 0.0, 0.0),
        PurePursuitTracker::new(1.0).expect("a valid lookahead"),
        NoAvoidance,
        TrackingSettings::default(),
    )
    .expect("the defaults are consistent");

    for dt in REJECTED {
        assert!(
            loop_.step(&path, dt).is_err(),
            "the loop accepted an interval of {dt}"
        );
    }
    assert!(loop_.step(&path, 0.1).is_ok());
}

#[test]
fn a_narrowed_band_is_the_one_that_is_enforced() {
    // The band is configuration, not a constant: a loop running at a
    // thousand hertz and one running at ten want different answers about
    // what a missed deadline looks like.
    let band = IntervalBand::new(0.009, 0.011).expect("a valid band");
    let mut controller = PidController::new(PidSettings {
        interval: band,
        ..PidSettings::default()
    })
    .expect("valid settings");

    assert!(controller.step(0.0, 1.0, 0.01).is_ok());
    // Inside the default band, outside this one.
    assert!(matches!(
        controller.step(0.0, 1.0, 0.1),
        Err(Error::OutOfRange { .. })
    ));
}
