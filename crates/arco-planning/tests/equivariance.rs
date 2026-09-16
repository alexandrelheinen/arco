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

//! FR-INV-05: planning in a moved frame gives the moved answer.
//!
//! The strongest metamorphic relation available to a sampling planner.
//! Nothing here needs to know what the right path is: rotating and
//! translating the map, the start, the goal and the sampler together must
//! move the answer by exactly that much, because every decision the
//! planner makes is a distance comparison and a rigid motion preserves
//! distances.
//!
//! Two settings have to be pinned for the relation to be testable rather
//! than merely true in principle. The metric scale is isotropic, since a
//! per-axis scale is not rotation invariant and the planner would be
//! measuring a different space after the turn. And the rewiring radius is
//! fixed, because the scheduled one is derived from the sampler's own
//! description of the volume it covers, which a transformed sampler
//! cannot state as an axis-aligned box.

use arco_core::Error;
use arco_core::protocols::Sampler;
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, RrtPlanner, RrtSettings, SamplerPolicy, SegmentPolicy, SstPlanner, SstSettings,
    SteererPolicy,
};

const STEP_SIZE: f64 = 2.0;
const BOUNDS: [(f64, f64); 2] = [(0.0, 50.0), (0.0, 50.0)];
const START: [f64; 2] = [2.0, 2.0];
const GOAL: [f64; 2] = [48.0, 48.0];
const REWIRE_RADIUS: f64 = 1.8;
const TOLERANCE: f64 = 1e-9;

/// A rotation about the origin followed by a translation.
#[derive(Debug, Clone, Copy)]
struct RigidMotion {
    angle: f64,
    shift: (f64, f64),
}

impl RigidMotion {
    fn apply(self, point: &[f64]) -> Vec<f64> {
        let (sine, cosine) = self.angle.sin_cos();
        let x = point.first().copied().unwrap_or_default();
        let y = point.get(1).copied().unwrap_or_default();
        vec![
            cosine.mul_add(x, -(sine * y)) + self.shift.0,
            sine.mul_add(x, cosine * y) + self.shift.1,
        ]
    }
}

/// Samples the original box, then moves the sample into the new frame.
///
/// Drawing in the original frame and mapping afterward is what keeps the
/// two runs consuming the generator identically. Sampling a box fitted
/// around the rotated region would draw different numbers and the
/// comparison would prove nothing.
#[derive(Debug)]
struct MovedSampler {
    motion: RigidMotion,
}

impl Sampler for MovedSampler {
    fn sample(&self, generator: &mut Pcg64) -> Result<Vec<f64>, Error> {
        let point: Vec<f64> = BOUNDS
            .iter()
            .map(|&(low, high)| low + generator.next_f64() * (high - low))
            .collect();
        Ok(self.motion.apply(&point))
    }
}

fn obstacle_field(seed: u64, count: usize) -> Vec<Vec<f64>> {
    let mut generator = Pcg64::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            vec![
                generator.next_f64().mul_add(42.0, 4.0),
                generator.next_f64().mul_add(42.0, 4.0),
            ]
        })
        .collect()
}

fn metric() -> CostPolicy {
    CostPolicy::Scaled {
        step_size: vec![STEP_SIZE, STEP_SIZE],
    }
}

fn steerer() -> SteererPolicy {
    SteererPolicy::Straight {
        step_size: vec![STEP_SIZE, STEP_SIZE],
    }
}

fn rrt_settings() -> RrtSettings {
    RrtSettings {
        max_samples: 2000,
        goal_tolerance: 1.5,
        fixed_rewire_radius: Some(REWIRE_RADIUS),
        early_stop: false,
        ..RrtSettings::default()
    }
}

fn sst_settings() -> SstSettings {
    SstSettings {
        max_samples: 2000,
        goal_tolerance: 1.5,
        early_stop: false,
        ..SstSettings::default()
    }
}

/// Asserts that `moved` is `original` under `motion`, point by point.
fn assert_is_image(original: &[Vec<f64>], moved: &[Vec<f64>], motion: RigidMotion) {
    assert_eq!(
        original.len(),
        moved.len(),
        "the moved frame produced a path of a different length"
    );
    for (index, (before, after)) in original.iter().zip(moved).enumerate() {
        let expected = motion.apply(before);
        for (axis, (wanted, got)) in expected.iter().zip(after).enumerate() {
            assert!(
                (wanted - got).abs() <= TOLERANCE,
                "waypoint {index} axis {axis}: expected {wanted}, got {got}"
            );
        }
    }
}

#[test]
fn moving_the_whole_problem_moves_the_rrt_path_with_it() {
    let motion = RigidMotion {
        angle: 0.7,
        shift: (-13.0, 41.5),
    };
    let points = obstacle_field(5, 120);
    let moved_points: Vec<Vec<f64>> = points.iter().map(|point| motion.apply(point)).collect();

    let here = RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: BOUNDS.to_vec(),
        },
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&points, 1.2).unwrap(),
        },
        metric(),
        rrt_settings(),
    );
    let there = RrtPlanner::new(
        SamplerPolicy::Custom(Box::new(MovedSampler { motion })),
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&moved_points, 1.2).unwrap(),
        },
        metric(),
        rrt_settings(),
    );

    let mut first = Pcg64::seed_from_u64(31);
    let mut second = Pcg64::seed_from_u64(31);
    let original = here.plan(&START, &GOAL, &mut first).unwrap();
    let moved = there
        .plan(&motion.apply(&START), &motion.apply(&GOAL), &mut second)
        .unwrap();

    let original_path = original.path().expect("this field is crossable");
    let moved_path = moved.path().expect("a rigid motion cannot block a path");
    assert_is_image(original_path, moved_path, motion);
    assert!(
        (original.cost().unwrap() - moved.cost().unwrap()).abs() <= TOLERANCE,
        "a rigid motion changed the path cost"
    );
}

#[test]
fn moving_the_whole_problem_moves_the_sst_path_with_it() {
    let motion = RigidMotion {
        angle: -2.1,
        shift: (7.25, -3.0),
    };
    let points = obstacle_field(6, 120);
    let moved_points: Vec<Vec<f64>> = points.iter().map(|point| motion.apply(point)).collect();

    let here = SstPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: BOUNDS.to_vec(),
        },
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&points, 1.2).unwrap(),
        },
        metric(),
        sst_settings(),
    );
    let there = SstPlanner::new(
        SamplerPolicy::Custom(Box::new(MovedSampler { motion })),
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&moved_points, 1.2).unwrap(),
        },
        metric(),
        sst_settings(),
    );

    let mut first = Pcg64::seed_from_u64(32);
    let mut second = Pcg64::seed_from_u64(32);
    let original = here.plan(&START, &GOAL, &mut first).unwrap();
    let moved = there
        .plan(&motion.apply(&START), &motion.apply(&GOAL), &mut second)
        .unwrap();

    let original_path = original.path().expect("this field is crossable");
    let moved_path = moved.path().expect("a rigid motion cannot block a path");
    assert_is_image(original_path, moved_path, motion);
}

#[test]
fn a_pure_translation_moves_the_path_with_it() {
    // A special case worth pinning on its own: a planner that reaches for
    // an absolute coordinate anywhere, an origin or a fixed bound, fails
    // this while a rotation test could have its disagreement waved off as
    // rounding.
    let motion = RigidMotion {
        angle: 0.0,
        shift: (100.0, -64.0),
    };
    let points = obstacle_field(7, 100);
    let moved_points: Vec<Vec<f64>> = points.iter().map(|point| motion.apply(point)).collect();

    let here = RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: BOUNDS.to_vec(),
        },
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&points, 1.2).unwrap(),
        },
        metric(),
        rrt_settings(),
    );
    let there = RrtPlanner::new(
        SamplerPolicy::Custom(Box::new(MovedSampler { motion })),
        steerer(),
        SegmentPolicy::Exact {
            occupancy: KdTreeOccupancy::new(&moved_points, 1.2).unwrap(),
        },
        metric(),
        rrt_settings(),
    );

    let mut first = Pcg64::seed_from_u64(33);
    let mut second = Pcg64::seed_from_u64(33);
    let original = here.plan(&START, &GOAL, &mut first).unwrap();
    let moved = there
        .plan(&motion.apply(&START), &motion.apply(&GOAL), &mut second)
        .unwrap();

    let original_path = original.path().expect("this field is crossable");
    let moved_path = moved.path().expect("a translation cannot block a path");
    assert_is_image(original_path, moved_path, motion);
}
