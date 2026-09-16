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

//! FR-RNG-02: the generator matches `numpy.random.default_rng` exactly.
//!
//! A seeded planner has to draw the values the Python implementation drew,
//! or every result recorded against a seed changes meaning. The fixtures
//! come from numpy itself, via `scripts/generate_rng_fixtures.py`.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions"
)]

use arco_core::rng::{Pcg64, SeedSequence};

const FIXTURES: &str = include_str!("fixtures/pcg64.json");

/// One captured seed and the draws numpy produced from it.
struct Case {
    seed: Vec<u32>,
    initial_state: u128,
    increment: u128,
    raw_u64: Vec<u64>,
    random_f64_bits: Vec<u64>,
}

/// Split a decimal seed into the little-endian `u32` words numpy uses.
///
/// Long division of the decimal digits by 2^32, most significant digit
/// first, so each remainder is one word and each quotient digit stays in
/// 0 to 9.
fn seed_words(decimal: &str) -> Vec<u32> {
    const BASE: u64 = 1 << 32;

    let mut digits: Vec<u8> = decimal.bytes().map(|byte| byte - b'0').collect();
    let mut words = Vec::new();

    while digits.iter().any(|&digit| digit != 0) {
        let mut remainder: u64 = 0;
        let mut quotient = Vec::with_capacity(digits.len());
        for &digit in &digits {
            let value = remainder * 10 + u64::from(digit);
            quotient.push(u8::try_from(value / BASE).unwrap());
            remainder = value % BASE;
        }
        words.push(u32::try_from(remainder).unwrap());
        while quotient.len() > 1 && quotient[0] == 0 {
            quotient.remove(0);
        }
        digits = quotient;
    }

    if words.is_empty() {
        words.push(0);
    }
    words
}

fn parse_hex_u128(text: &str) -> u128 {
    u128::from_str_radix(text.trim_start_matches("0x"), 16).unwrap()
}

fn parse_hex_u64(text: &str) -> u64 {
    u64::from_str_radix(text.trim_start_matches("0x"), 16).unwrap()
}

fn cases() -> Vec<Case> {
    let parsed: serde_json::Value = serde_json::from_str(FIXTURES).unwrap();
    parsed["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| Case {
            seed: seed_words(case["seed"].as_str().unwrap()),
            initial_state: parse_hex_u128(case["initial_state"].as_str().unwrap()),
            increment: parse_hex_u128(case["increment"].as_str().unwrap()),
            raw_u64: case["raw_u64"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| parse_hex_u64(value.as_str().unwrap()))
                .collect(),
            // Bit patterns rather than decimals: serde_json's float parser
            // and Rust's own disagree by one unit in the last place on at
            // least one of these values, and this test is about the bits.
            random_f64_bits: case["random_f64_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| parse_hex_u64(value.as_str().unwrap()))
                .collect(),
        })
        .collect()
}

#[test]
fn seeding_reproduces_the_numpy_state_and_increment() {
    for case in cases() {
        let generator = Pcg64::from_seed_sequence(&SeedSequence::new(&case.seed));
        assert_eq!(
            generator.state(),
            case.initial_state,
            "state mismatch for seed words {:?}",
            case.seed
        );
        assert_eq!(
            generator.increment(),
            case.increment,
            "increment mismatch for seed words {:?}",
            case.seed
        );
    }
}

#[test]
fn raw_draws_match_numpy_value_for_value() {
    for case in cases() {
        let mut generator = Pcg64::from_seed_sequence(&SeedSequence::new(&case.seed));
        for (index, &expected) in case.raw_u64.iter().enumerate() {
            assert_eq!(
                generator.next_u64(),
                expected,
                "raw draw {index} differs for seed words {:?}",
                case.seed
            );
        }
    }
}

#[test]
fn unit_interval_draws_match_numpy_bit_for_bit() {
    for case in cases() {
        let mut generator = Pcg64::from_seed_sequence(&SeedSequence::new(&case.seed));
        for (index, &expected) in case.random_f64_bits.iter().enumerate() {
            let produced = generator.next_f64();
            assert_eq!(
                produced.to_bits(),
                expected,
                "double {index} differs for seed words {:?}: {produced} against {}",
                case.seed,
                f64::from_bits(expected)
            );
        }
    }
}

#[test]
fn a_seed_replays_identically() {
    // FR-RNG-01, which is weaker than FR-RNG-02 but is the property a
    // caller actually relies on.
    let first: Vec<u64> = {
        let mut generator = Pcg64::seed_from_u64(20_260_916);
        (0..32).map(|_| generator.next_u64()).collect()
    };
    let second: Vec<u64> = {
        let mut generator = Pcg64::seed_from_u64(20_260_916);
        (0..32).map(|_| generator.next_u64()).collect()
    };
    assert_eq!(first, second);
}

#[test]
fn unit_interval_draws_stay_in_range() {
    let mut generator = Pcg64::seed_from_u64(7);
    for _ in 0..10_000 {
        let value = generator.next_f64();
        assert!((0.0..1.0).contains(&value), "{value} left [0, 1)");
    }
}
