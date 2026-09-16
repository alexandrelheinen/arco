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

//! FR-INV-09: a command lies inside the box and is reachable from the last.
//!
//! Deviation A-09 is what these prove. The two limiters are asserted
//! apart because they fail apart: a magnitude clamp opens the loop, a rate
//! limit adds phase lag, and a caller told only that something was limited
//! cannot tell which of those it is living with.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_control::limits::{CommandConditioner, CommandLimits, IntervalBand};
use arco_core::Error;
use arco_core::protocols::Command;

fn limits() -> CommandLimits {
    CommandLimits {
        max_speed: 2.0,
        min_speed: -0.5,
        max_turn_rate: 1.0,
        max_speed_rate: 4.0,
        max_turn_rate_change: 2.0,
        interval: IntervalBand::default(),
    }
}

fn conditioner() -> CommandConditioner {
    CommandConditioner::new(limits()).expect("the fixture limits are consistent")
}

#[test]
fn an_applied_command_always_lies_inside_the_box() {
    // FR-INV-09, first half. Swept rather than spot-checked, because the
    // clamp is the one thing here that has to hold for every input.
    let mut conditioner = conditioner();
    let mut requested_speed = -10.0_f64;
    while requested_speed <= 10.0 {
        let applied = conditioner
            .apply(
                Command {
                    speed: requested_speed,
                    turn_rate: requested_speed,
                },
                0.1,
            )
            .expect("a finite command at a valid interval is accepted");
        assert!(
            applied.speed >= limits().min_speed - 1e-12
                && applied.speed <= limits().max_speed + 1e-12,
            "speed {} left the box",
            applied.speed
        );
        assert!(
            applied.turn_rate.abs() <= limits().max_turn_rate + 1e-12,
            "turn rate {} left the box",
            applied.turn_rate
        );
        requested_speed += 0.37;
    }
}

#[test]
fn an_applied_command_is_reachable_from_the_previous_one() {
    // FR-INV-09, second half. The step between consecutive applied
    // commands never exceeds the rate limit times the interval.
    let mut conditioner = conditioner();
    let dt = 0.05;
    let mut previous = None;
    for step in 0..40_i32 {
        let requested = Command {
            // Alternates hard between the extremes, which is the input a
            // rate limiter exists for.
            speed: if step % 2 == 0 { 5.0 } else { -5.0 },
            turn_rate: if step % 2 == 0 { -3.0 } else { 3.0 },
        };
        let applied = conditioner.apply(requested, dt).expect("valid step");
        if let Some(previous) = previous {
            let previous: Command = previous;
            assert!(
                (applied.speed - previous.speed).abs() <= limits().max_speed_rate * dt + 1e-12,
                "speed jumped from {} to {}",
                previous.speed,
                applied.speed
            );
            assert!(
                (applied.turn_rate - previous.turn_rate).abs()
                    <= limits().max_turn_rate_change * dt + 1e-12,
                "turn rate jumped from {} to {}",
                previous.turn_rate,
                applied.turn_rate
            );
        }
        previous = Some(applied);
    }
}

#[test]
fn the_first_command_is_clamped_but_not_rate_limited() {
    // There is nothing to have changed from, so limiting the first command
    // by rate would mean every run begins by ramping from zero whatever
    // the vehicle was actually doing.
    let mut conditioner = conditioner();
    let applied = conditioner
        .apply(
            Command {
                speed: 100.0,
                turn_rate: 0.0,
            },
            0.01,
        )
        .expect("valid step");
    assert!((applied.speed - limits().max_speed).abs() < 1e-12);
    assert_eq!(conditioner.report().magnitude_steps, 1);
    assert_eq!(conditioner.report().rate_steps, 0);
}

#[test]
fn the_two_saturations_are_reported_apart() {
    // The reason they are separate fields. A magnitude clamp exhausts
    // authority; a rate limit injects lag. NASA TN D-7900 found the rate
    // limits the more damaging of the two on the YF-12, so collapsing them
    // into one flag throws away the part that mattered.
    let mut conditioner = conditioner();
    // Settle at zero so the rate limiter has a previous command.
    conditioner
        .apply(
            Command {
                speed: 0.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("valid step");

    // Inside the box, but far from the previous command: rate only.
    let applied = conditioner
        .apply(
            Command {
                speed: 2.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("valid step");
    assert!(applied.speed < 2.0, "the rate limiter did not bite");
    assert_eq!(conditioner.report().magnitude_steps, 0);
    assert_eq!(conditioner.report().rate_steps, 1);

    // Outside the box as well: both.
    let before = conditioner.report();
    conditioner
        .apply(
            Command {
                speed: 50.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("valid step");
    let after = conditioner.report();
    assert_eq!(after.magnitude_steps, before.magnitude_steps + 1);
    assert_eq!(after.rate_steps, before.rate_steps + 1);
}

#[test]
fn the_report_accumulates_what_was_asked_for_and_not_given() {
    // Reports, never decides. The integral is what an integrating system
    // reads to judge how long it has been running open loop; no primary
    // source says how much is too much, so this type does not guess.
    let mut conditioner = conditioner();
    for _ in 0..10 {
        conditioner
            .apply(
                Command {
                    speed: 12.0,
                    turn_rate: 0.0,
                },
                0.1,
            )
            .expect("valid step");
    }
    let report = conditioner.report();
    assert!(report.saturated());
    assert!(
        report.speed_excess > 90.0,
        "ten steps asking for 12 against a limit of 2 lost {}",
        report.speed_excess
    );
    assert!(report.turn_rate_excess.abs() < 1e-12);
}

#[test]
fn an_unlimited_conditioner_passes_everything_through() {
    // The default limits do not bite, so a caller who never asked for a
    // limit gets exactly what it requested. Deviation A-09 adds the
    // mechanism, not a limit nobody chose.
    let mut conditioner =
        CommandConditioner::new(CommandLimits::default()).expect("defaults are consistent");
    for speed in [-1e6, -1.0, 0.0, 1.0, 1e6] {
        let requested = Command {
            speed,
            turn_rate: -speed,
        };
        let applied = conditioner.apply(requested, 0.001).expect("valid step");
        assert_eq!(applied, requested);
    }
    assert!(!conditioner.report().saturated());
}

#[test]
fn resetting_forgets_the_previous_command_and_the_counters() {
    let mut conditioner = conditioner();
    conditioner
        .apply(
            Command {
                speed: 99.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("valid step");
    assert!(conditioner.previous().is_some());
    assert!(conditioner.report().saturated());

    conditioner.reset();
    assert!(conditioner.previous().is_none());
    assert!(!conditioner.report().saturated());
}

#[test]
fn a_non_finite_command_is_rejected_rather_than_clamped() {
    // FR-SAFE-07. A NaN clamps to a NaN and reaches the actuator, and
    // every comparison downstream of it is false.
    let mut conditioner = conditioner();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            conditioner.apply(
                Command {
                    speed: value,
                    turn_rate: 0.0
                },
                0.1
            ),
            Err(Error::NotFinite { .. })
        ));
        assert!(matches!(
            conditioner.apply(
                Command {
                    speed: 0.0,
                    turn_rate: value
                },
                0.1
            ),
            Err(Error::NotFinite { .. })
        ));
    }
}

#[test]
fn an_inconsistent_limit_set_is_rejected_at_construction() {
    let inverted = CommandLimits {
        min_speed: 3.0,
        max_speed: 1.0,
        ..limits()
    };
    assert!(matches!(
        CommandConditioner::new(inverted),
        Err(Error::OutOfRange { .. })
    ));

    let negative_rate = CommandLimits {
        max_speed_rate: -1.0,
        ..limits()
    };
    assert!(matches!(
        CommandConditioner::new(negative_rate),
        Err(Error::OutOfRange { .. })
    ));

    let nan = CommandLimits {
        max_turn_rate: f64::NAN,
        ..limits()
    };
    assert!(matches!(
        CommandConditioner::new(nan),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn a_zero_rate_limit_freezes_the_command() {
    // Degenerate but legal, and worth pinning: a rate limit of zero means
    // the command cannot change, which is different from rejecting it.
    let mut conditioner = CommandConditioner::new(CommandLimits {
        max_speed_rate: 0.0,
        max_turn_rate_change: 0.0,
        ..limits()
    })
    .expect("a zero rate limit is legal");

    let first = conditioner
        .apply(
            Command {
                speed: 1.0,
                turn_rate: 0.5,
            },
            0.1,
        )
        .expect("valid step");
    let second = conditioner
        .apply(
            Command {
                speed: 2.0,
                turn_rate: -0.5,
            },
            0.1,
        )
        .expect("valid step");
    assert_eq!(first, second);
}
