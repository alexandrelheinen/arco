//! numpy's `SeedSequence`, which turns arbitrary entropy into seed words.
//!
//! The constants and the mixing order below reproduce numpy's
//! implementation exactly, because `FR-RNG-02` is a claim about matching
//! its output rather than about producing good entropy independently.

/// Number of 32 bit words in the mixing pool.
const POOL_SIZE: usize = 4;

/// Initial hash constant for entropy mixing.
const INIT_A: u32 = 0x43b0_d7e5;
/// Multiplier advancing the hash constant during mixing.
const MULT_A: u32 = 0x931e_8875;
/// Initial hash constant for state generation.
const INIT_B: u32 = 0x8b51_f9dd;
/// Multiplier advancing the hash constant during state generation.
const MULT_B: u32 = 0x58f3_8ded;
/// Left multiplier of the two-word mixing step.
const MIX_MULT_L: u32 = 0xca01_f9dd;
/// Right multiplier of the two-word mixing step.
const MIX_MULT_R: u32 = 0x4973_f715;
/// Xor-shift distance, half the width of a word.
const XSHIFT: u32 = 16;

/// Hashes `value`, advancing `hash_const` as a side effect.
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut mixed = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    mixed = mixed.wrapping_mul(*hash_const);
    mixed ^ (mixed >> XSHIFT)
}

/// Combines two pool words into one.
fn mix(left: u32, right: u32) -> u32 {
    let combined = MIX_MULT_L
        .wrapping_mul(left)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(right));
    combined ^ (combined >> XSHIFT)
}

/// Entropy mixer producing the seed words a bit generator initializes from.
///
/// Construct one from the little-endian `u32` words of a seed, then ask it
/// for as much seed material as the generator needs.
///
/// # Examples
///
/// ```
/// use arco_core::rng::SeedSequence;
///
/// let sequence = SeedSequence::new(&[0]);
/// assert_eq!(sequence.generate_state_u64(4).len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct SeedSequence {
    /// The mixed entropy pool.
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// Mixes `entropy`, given as little-endian `u32` words.
    ///
    /// # Arguments
    ///
    /// * `entropy` - Seed words, least significant first. An empty slice
    ///   is treated as a single zero word, matching a seed of zero.
    #[must_use]
    pub fn new(entropy: &[u32]) -> Self {
        let mut hash_const = INIT_A;
        let mut pool = [0_u32; POOL_SIZE];

        for (index, slot) in pool.iter_mut().enumerate() {
            let word = entropy.get(index).copied().unwrap_or(0);
            *slot = hashmix(word, &mut hash_const);
        }

        // Each pool word is mixed into every other one. The source word is
        // never written during its own pass, so reading it once is the same
        // as re-reading it per destination.
        for source in 0..POOL_SIZE {
            let source_word = pool.get(source).copied().unwrap_or_default();
            for (destination, slot) in pool.iter_mut().enumerate() {
                if source != destination {
                    let hashed = hashmix(source_word, &mut hash_const);
                    *slot = mix(*slot, hashed);
                }
            }
        }

        for &word in entropy.iter().skip(POOL_SIZE) {
            for slot in &mut pool {
                let hashed = hashmix(word, &mut hash_const);
                *slot = mix(*slot, hashed);
            }
        }

        Self { pool }
    }

    /// Produces `word_count` 32 bit words of seed material.
    #[must_use]
    pub fn generate_state_u32(&self, word_count: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        let mut state = Vec::with_capacity(word_count);

        for &word in self.pool.iter().cycle().take(word_count) {
            let mut value = word ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            state.push(value ^ (value >> XSHIFT));
        }

        state
    }

    /// Produces `word_count` 64 bit words of seed material.
    ///
    /// Each 64 bit word pairs two 32 bit words little-endian, which is how
    /// numpy reinterprets the same buffer.
    #[must_use]
    pub fn generate_state_u64(&self, word_count: usize) -> Vec<u64> {
        let words = self.generate_state_u32(word_count.saturating_mul(2));
        let mut narrow = words.into_iter();
        let mut wide = Vec::with_capacity(word_count);
        while let (Some(low), Some(high)) = (narrow.next(), narrow.next()) {
            wide.push(u64::from(low) | (u64::from(high) << 32));
        }
        wide
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_seed_mixes_as_a_single_zero_word() {
        assert_eq!(SeedSequence::new(&[]).pool, SeedSequence::new(&[0]).pool);
    }

    #[test]
    fn generating_more_words_extends_rather_than_replaces() {
        let sequence = SeedSequence::new(&[42]);
        let short = sequence.generate_state_u32(4);
        let long = sequence.generate_state_u32(8);
        assert_eq!(short, long[..4]);
    }

    #[test]
    fn sixty_four_bit_words_pair_the_thirty_two_bit_ones() {
        let sequence = SeedSequence::new(&[7]);
        let narrow = sequence.generate_state_u32(4);
        let wide = sequence.generate_state_u64(2);
        assert_eq!(wide[0], u64::from(narrow[0]) | (u64::from(narrow[1]) << 32));
        assert_eq!(wide[1], u64::from(narrow[2]) | (u64::from(narrow[3]) << 32));
        assert_eq!(wide.len(), 2);
    }
}
