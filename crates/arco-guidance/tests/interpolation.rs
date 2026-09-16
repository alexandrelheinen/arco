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

//! What smoothing a polyline is allowed to do to it.
//!
//! The window arithmetic is pinned against means worked out by hand, since
//! the truncation at each end is the part a reader cannot check by
//! inspection and the part a rewrite would get wrong first.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_core::Error;
use arco_guidance::interpolation::{BSplineInterpolator, Interpolator, MovingAverageInterpolator};

/// A smoother of the given strength, or a panic if the test asked for one
/// that cannot exist.
fn smoother(iterations: usize, window: usize) -> MovingAverageInterpolator {
    MovingAverageInterpolator::new(iterations, window).expect("a valid smoother")
}

/// A zigzag of unit amplitude about the first axis.
fn zigzag(count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|index| {
            let x = f64::from(u32::try_from(index).unwrap_or(u32::MAX));
            (x, if index % 2 == 0 { -1.0 } else { 1.0 })
        })
        .collect()
}

/// The largest excursion away from the first axis, ends excluded.
///
/// The three waypoints nearest each end are left out: the endpoints are
/// preserved exactly and hold their neighbors out with them, so including
/// them would measure the boundary rather than the smoothing.
fn wiggle(path: &[(f64, f64)]) -> f64 {
    path.get(3..path.len().saturating_sub(3))
        .unwrap_or_default()
        .iter()
        .fold(0.0_f64, |worst, &(_, y)| worst.max(y.abs()))
}

/// Whether two paths agree waypoint for waypoint.
fn close(left: &[(f64, f64)], right: &[(f64, f64)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12)
}

// --------------------------------------------------- moving average ----

#[test]
fn one_pass_of_a_three_wide_window_is_the_mean_of_the_three() {
    let smoothed = smoother(1, 3)
        .interpolate(&[(0.0, 0.0), (1.0, 3.0), (2.0, 0.0)])
        .expect("a finite path");
    assert!(
        close(&smoothed, &[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),
        "{smoothed:?}"
    );
}

#[test]
fn a_window_reaching_past_the_end_is_truncated_rather_than_shifted() {
    // Five waypoints under a five-wide window: the second point averages
    // the four that exist rather than four plus a mirrored invention, and
    // the middle one averages all five. Working the means out by hand is
    // the only way to catch a window that quietly slides inward to keep
    // its width.
    let smoothed = smoother(1, 5)
        .interpolate(&[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0)])
        .expect("a finite path");
    let expected = [(0.0, 0.0), (1.5, 0.5), (2.0, 0.4), (2.5, 0.5), (4.0, 0.0)];
    assert!(close(&smoothed, &expected), "{smoothed:?}");
}

#[test]
fn the_endpoints_survive_every_pass() {
    let path = [(0.0, 0.0), (1.0, 2.0), (2.0, -2.0), (3.0, 0.5), (4.0, 0.0)];
    for iterations in 1..=5 {
        let smoothed = smoother(iterations, 3)
            .interpolate(&path)
            .expect("a finite path");
        assert_eq!(
            smoothed.len(),
            path.len(),
            "{iterations} passes changed the length"
        );
        assert_eq!(
            smoothed.first(),
            path.first(),
            "{iterations} passes moved the start"
        );
        assert_eq!(
            smoothed.last(),
            path.last(),
            "{iterations} passes moved the end"
        );
    }
}

#[test]
fn a_straight_line_is_left_where_it_was() {
    // The mean of evenly spaced collinear points is the point itself, so
    // smoothing a straight path is the identity. A filter that fails this
    // is dragging the path somewhere, and on a straight run there is
    // nowhere it could honestly drag it to.
    let path: Vec<(f64, f64)> = (0..8)
        .map(|index| (f64::from(u32::try_from(index).unwrap_or(0)), 0.0))
        .collect();
    let smoothed = smoother(2, 3).interpolate(&path).expect("a finite path");
    assert!(close(&smoothed, &path), "{smoothed:?}");
}

#[test]
fn smoothing_shrinks_a_wiggle_and_more_passes_shrink_it_further() {
    let path = zigzag(12);
    let once = smoother(1, 3).interpolate(&path).expect("a finite path");
    let thrice = smoother(3, 3).interpolate(&path).expect("a finite path");
    assert!(wiggle(&once) < wiggle(&path), "{}", wiggle(&once));
    assert!(wiggle(&thrice) < wiggle(&once), "{}", wiggle(&thrice));
}

#[test]
fn a_wider_window_shrinks_a_wiggle_further_than_a_narrow_one() {
    let path = zigzag(16);
    let narrow = smoother(1, 3).interpolate(&path).expect("a finite path");
    let wide = smoother(1, 7).interpolate(&path).expect("a finite path");
    assert!(
        wiggle(&wide) < wiggle(&narrow),
        "{} against {}",
        wiggle(&wide),
        wiggle(&narrow)
    );
}

#[test]
fn no_waypoint_leaves_the_span_the_path_already_covered() {
    // Every smoothed point is a mean of points that were already there, so
    // the result is inside the original's bounding box however many passes
    // run. A path that grew outward would be one the clearance check ahead
    // of it had never seen.
    let path = zigzag(20);
    let smoothed = smoother(6, 5).interpolate(&path).expect("a finite path");
    let bound = |points: &[(f64, f64)]| {
        points.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(low_x, high_x, low_y, high_y), &(x, y)| {
                (low_x.min(x), high_x.max(x), low_y.min(y), high_y.max(y))
            },
        )
    };
    let (low_x, high_x, low_y, high_y) = bound(&path);
    let (inner_low_x, inner_high_x, inner_low_y, inner_high_y) = bound(&smoothed);
    assert!(inner_low_x >= low_x - 1e-12 && inner_high_x <= high_x + 1e-12);
    assert!(inner_low_y >= low_y - 1e-12 && inner_high_y <= high_y + 1e-12);
}

#[test]
fn a_path_with_no_interior_passes_through_untouched() {
    let filter = smoother(4, 3);
    assert!(filter.interpolate(&[]).expect("a finite path").is_empty());
    for path in [vec![(1.0, 2.0)], vec![(0.0, 0.0), (1.0, 1.0)]] {
        let smoothed = filter.interpolate(&path).expect("a finite path");
        assert!(close(&smoothed, &path), "{smoothed:?}");
    }
}

#[test]
fn a_window_that_has_no_center_is_refused() {
    for window in [0, 1, 2, 4, 10] {
        assert!(
            matches!(
                MovingAverageInterpolator::new(1, window),
                Err(Error::OutOfRange { .. })
            ),
            "a window of {window} was accepted"
        );
    }
    assert!(matches!(
        MovingAverageInterpolator::new(0, 3),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn the_smoother_keeps_what_it_was_built_with() {
    let filter = smoother(4, 7);
    assert_eq!(filter.iterations(), 4);
    assert_eq!(filter.window(), 7);
}

#[test]
fn a_waypoint_that_is_not_a_real_number_is_refused() {
    // One NaN would spread across every window that touches it, so a path
    // smoothed without this check comes back mostly NaN and the first
    // thing to notice is the vehicle.
    let filter = smoother(1, 3);
    for path in [
        vec![(0.0, 0.0), (f64::NAN, 1.0), (2.0, 0.0)],
        vec![(0.0, 0.0), (1.0, f64::INFINITY), (2.0, 0.0)],
        vec![(f64::NAN, 0.0), (1.0, 0.0)],
    ] {
        assert!(
            matches!(filter.interpolate(&path), Err(Error::NotFinite { .. })),
            "{path:?}"
        );
    }
}

// ---------------------------------------------------------- b-spline ----

#[test]
fn the_b_spline_returns_the_path_it_was_given() {
    // `arco.guidance.interpolation.bspline` is a placeholder that returns
    // its argument, and the port carries that across rather than inventing
    // a curve every existing caller would suddenly be following.
    let path = zigzag(6);
    let smoothed = BSplineInterpolator::new(3)
        .expect("a valid degree")
        .interpolate(&path)
        .expect("a finite path");
    assert!(close(&smoothed, &path), "{smoothed:?}");
}

#[test]
fn the_b_spline_keeps_the_degree_it_was_built_with() {
    let interpolator = BSplineInterpolator::new(5).expect("a valid degree");
    assert_eq!(interpolator.degree(), 5);
    assert!(matches!(
        BSplineInterpolator::new(0),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn the_b_spline_refuses_a_waypoint_that_is_not_a_real_number() {
    // It changes nothing about the path, and still rejects what the other
    // interpolators reject, so swapping one for the other cannot turn a
    // refusal into a silent pass.
    let interpolator = BSplineInterpolator::new(3).expect("a valid degree");
    assert!(matches!(
        interpolator.interpolate(&[(0.0, 0.0), (1.0, f64::NAN)]),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn both_interpolators_answer_through_the_trait() {
    // The seam is the trait: a caller holding one of these does not know
    // which it has, which is what lets a scene swap the smoother out.
    let path = zigzag(9);
    let interpolators: [&dyn Interpolator; 2] = [
        &smoother(2, 3),
        &BSplineInterpolator::new(3).expect("a valid degree"),
    ];
    for interpolator in interpolators {
        let smoothed = interpolator.interpolate(&path).expect("a finite path");
        assert_eq!(smoothed.len(), path.len());
        assert_eq!(smoothed.first(), path.first());
        assert_eq!(smoothed.last(), path.last());
    }
}
