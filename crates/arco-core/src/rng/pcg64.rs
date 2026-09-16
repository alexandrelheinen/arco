//! PCG64 with the XSL-RR output function, as numpy implements it.

use super::SeedSequence;

/// The PCG64 multiplier, fixed by the algorithm.
const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// Scale taking a 53 bit integer to a double in `[0, 1)`.
const UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Truncates to the low 64 bits.
///
/// The PCG64 output function is defined in terms of this truncation, so
/// the discarded half is intentional rather than an overflow.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the PCG64 output function is defined as a truncation of the 128 bit state"
)]
const fn low_half(value: u128) -> u64 {
    value as u64
}

/// PCG64 generator producing the same stream as `numpy.random.default_rng`.
///
/// # Examples
///
/// ```
/// use arco_core::rng::Pcg64;
///
/// let mut generator = Pcg64::seed_from_u64(0);
/// assert_eq!(generator.next_u64(), 0xa30f_ebcf_d9c2_825f);
/// ```
#[derive(Debug, Clone)]
pub struct Pcg64 {
    /// The 128 bit LCG state.
    state: u128,
    /// The odd increment selecting this generator's stream.
    increment: u128,
}

impl Pcg64 {
    /// Builds a generator from mixed seed material.
    ///
    /// Consumes four 64 bit words, the first pair seeding the state and
    /// the second pair selecting the stream, which is how numpy seeds its
    /// own PCG64.
    #[must_use]
    pub fn from_seed_sequence(sequence: &SeedSequence) -> Self {
        let words = sequence.generate_state_u64(4);
        let pack = |high: u64, low: u64| (u128::from(high) << 64) | u128::from(low);

        // The words pair most significant first, which is the opposite of
        // how the 32 bit words inside each of them pair. generate_state_u64
        // always returns the requested count, so the fallback is unreachable
        // and exists only to keep this total.
        let [state_high, state_low, increment_high, increment_low] =
            <[u64; 4]>::try_from(words.as_slice()).unwrap_or_default();
        let seed_state = pack(state_high, state_low);
        let seed_increment = pack(increment_high, increment_low);

        let increment = (seed_increment << 1) | 1;
        let mut generator = Self {
            state: 0,
            increment,
        };

        generator.step();
        generator.state = generator.state.wrapping_add(seed_state);
        generator.step();

        generator
    }

    /// Builds a generator from a single integer seed.
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "splitting a u64 into its two little-endian u32 words"
        )]
        let words = [seed as u32, (seed >> 32) as u32];
        let entropy: &[u32] = if words[1] == 0 { &words[..1] } else { &words };
        Self::from_seed_sequence(&SeedSequence::new(entropy))
    }

    /// The current 128 bit internal state.
    #[must_use]
    pub const fn state(&self) -> u128 {
        self.state
    }

    /// The 128 bit increment selecting this generator's stream.
    #[must_use]
    pub const fn increment(&self) -> u128 {
        self.increment
    }

    /// Advances the underlying linear congruential generator by one step.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// Draws the next raw 64 bit value.
    ///
    /// Steps first and then applies the output function, matching numpy.
    pub fn next_u64(&mut self) -> u64 {
        self.step();

        // XSL-RR: fold the two halves together, then rotate by the top six
        // bits of the state.
        let folded = low_half(self.state >> 64) ^ low_half(self.state);
        let rotation = u32::try_from(self.state >> 122).unwrap_or_default();
        folded.rotate_right(rotation)
    }

    /// Draws the next double in `[0, 1)`.
    ///
    /// Takes the top 53 bits, which is the only conversion that gives
    /// every representable double in the interval an equal chance.
    pub fn next_f64(&mut self) -> f64 {
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "shifting right by 11 bounds the value to 2^53 - 1, which a double represents exactly, so the conversion is lossless"
        )]
        let significand = (self.next_u64() >> 11) as f64;
        significand * UNIT_SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_increment_is_always_odd() {
        for seed in 0..64_u64 {
            assert_eq!(Pcg64::seed_from_u64(seed).increment() & 1, 1);
        }
    }

    #[test]
    fn different_seeds_give_different_streams() {
        let mut first = Pcg64::seed_from_u64(1);
        let mut second = Pcg64::seed_from_u64(2);
        assert_ne!(first.next_u64(), second.next_u64());
    }

    #[test]
    fn stepping_changes_the_state() {
        let mut generator = Pcg64::seed_from_u64(0);
        let before = generator.state();
        let _ = generator.next_u64();
        assert_ne!(generator.state(), before);
    }
}
