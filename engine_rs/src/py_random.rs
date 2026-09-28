//! CPython-compatible deterministic pseudo-random number generation.
//!
//! Kaggriculture seeds one fresh `random.Random` at each day boundary.  This
//! module implements the integer-seed path used by CPython 3.x, including the
//! MT19937 array initializer, the 53-bit `random()` float construction, and the
//! `getrandbits` rejection sampler used by `randrange`, `choice`, and `shuffle`.
//!
//! Python integers are unbounded, so [`PyRandom::from_seed_decimal`] and
//! [`PyRandom::from_seed_words_le`] cover seeds wider than Rust primitives.
//! Integer signs are intentionally discarded during seeding: CPython seeds
//! MT19937 from the absolute value of an `int`.
//! Zero-width `getrandbits(0)` follows CPython 3.9+ (return zero without advancing);
//! earlier 3.x releases rejected zero.  The checked vectors use CPython 3.11.15.
//!
//! This is deliberately the **integer** seed surface.  Decimal text is only a
//! transport for an already-decided Python integer; it does not implement the
//! SHA-512 preprocessing used when `Random.seed(version=2)` receives Python
//! `str`/`bytes`.  `None`, float seeds, Gaussian caching, and state import/export
//! are likewise outside the Kaggriculture reset path.

use std::error::Error;
use std::fmt;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// An unsigned Python-style integer returned by `getrandbits`.
///
/// Limbs are normalized little-endian base-2^32 words.  Zero has no limbs,
/// matching the value semantics of Python's integer zero rather than retaining
/// the requested allocation width.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyUInt {
    words_le: Vec<u32>,
}

impl PyUInt {
    fn from_words_le(mut words_le: Vec<u32>) -> Self {
        while words_le.last() == Some(&0) {
            words_le.pop();
        }
        Self { words_le }
    }

    /// Significant base-2^32 words, least-significant first.
    pub fn words_le(&self) -> &[u32] {
        &self.words_le
    }

    /// Return the value as `u64`, or `None` if it does not fit.
    pub fn to_u64(&self) -> Option<u64> {
        match self.words_le.as_slice() {
            [] => Some(0),
            [low] => Some(u64::from(*low)),
            [low, high] => Some(u64::from(*low) | (u64::from(*high) << 32)),
            _ => None,
        }
    }

    /// Lowercase hexadecimal using Python's `hex()` spelling.
    pub fn to_hex_string(&self) -> String {
        let Some((&high, rest)) = self.words_le.split_last() else {
            return "0x0".to_string();
        };
        let mut rendered = format!("0x{high:x}");
        for word in rest.iter().rev() {
            use fmt::Write as _;
            write!(&mut rendered, "{word:08x}").expect("writing to String cannot fail");
        }
        rendered
    }
}

/// Invalid textual representation of an integer seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedError {
    message: &'static str,
}

impl SeedError {
    fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl fmt::Display for SeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for SeedError {}

/// Domain errors for Python sequence/range helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomError {
    /// Python's `_randbelow` caller must supply a positive bound.
    EmptyRange,
    /// Python's `choice` raises `IndexError` for an empty sequence.
    EmptySequence,
    /// A fixed-width conversion was requested for more than 64 random bits.
    BitsTooWide,
}

impl fmt::Display for RandomError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyRange => "empty range for randrange()",
            Self::EmptySequence => "cannot choose from an empty sequence",
            Self::BitsTooWide => "requested random bits do not fit in u64",
        })
    }
}

impl Error for RandomError {}

/// State-compatible subset of CPython 3.x's `random.Random` for integer seeds.
#[derive(Clone)]
pub struct PyRandom {
    state: [u32; N],
    index: usize,
}

impl fmt::Debug for PyRandom {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PyRandom")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl PyRandom {
    /// Seed from a signed 64-bit integer through Python's absolute-value path.
    pub fn from_seed_i64(seed: i64) -> Self {
        Self::from_seed_i128(i128::from(seed))
    }

    /// Seed from a signed 128-bit integer through Python's absolute-value path.
    pub fn from_seed_i128(seed: i128) -> Self {
        let magnitude = seed.unsigned_abs();
        let mut words = Vec::with_capacity(4);
        let mut remaining = magnitude;
        while remaining != 0 {
            words.push(remaining as u32);
            remaining >>= 32;
        }
        Self::from_seed_words_le(&words)
    }

    /// Seed from an arbitrarily large base-10 integer.
    ///
    /// Leading/trailing whitespace and one leading ASCII sign are accepted.
    /// The magnitude is used for both positive and negative values, as it is by
    /// CPython's C-level integer seed implementation.
    pub fn from_seed_decimal(seed: &str) -> Result<Self, SeedError> {
        let mut digits = seed.trim();
        if let Some(rest) = digits.strip_prefix(['+', '-']) {
            digits = rest;
        }
        if digits.is_empty() {
            return Err(SeedError::new("integer seed contains no digits"));
        }

        let mut words = vec![0_u32];
        for byte in digits.bytes() {
            if !byte.is_ascii_digit() {
                return Err(SeedError::new("integer seed contains a non-decimal digit"));
            }
            multiply_add_words(&mut words, 10, u32::from(byte - b'0'));
        }
        trim_high_zero_words(&mut words);
        Ok(Self::from_seed_words_le(&words))
    }

    /// Seed from the absolute integer's base-2^32 words, least-significant first.
    ///
    /// Empty/all-zero input is canonicalized to the one-word zero key CPython
    /// passes into MT19937's array initializer.
    pub fn from_seed_words_le(words_le: &[u32]) -> Self {
        let mut key = words_le.to_vec();
        trim_high_zero_words(&mut key);
        if key.is_empty() {
            key.push(0);
        }

        let mut random = Self {
            state: [0; N],
            index: N,
        };
        random.init_by_array(&key);
        random
    }

    /// CPython's `random()` result: an exactly representable 53-bit fraction.
    pub fn random(&mut self) -> f64 {
        let high = u64::from(self.gen_u32() >> 5);
        let low = u64::from(self.gen_u32() >> 6);
        ((high << 26) | low) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// CPython-compatible `getrandbits(k)` for arbitrary `k`.
    pub fn getrandbits(&mut self, bits: usize) -> PyUInt {
        if bits == 0 {
            return PyUInt::default();
        }
        if bits <= 32 {
            return PyUInt::from_words_le(vec![self.gen_u32() >> (32 - bits)]);
        }

        let word_count = bits.div_ceil(32);
        let mut words = Vec::with_capacity(word_count);
        let mut remaining = bits;
        for _ in 0..word_count {
            let take = remaining.min(32);
            let mut word = self.gen_u32();
            if take < 32 {
                word >>= 32 - take;
            }
            words.push(word);
            remaining -= take;
        }
        PyUInt::from_words_le(words)
    }

    /// Fixed-width convenience wrapper around [`Self::getrandbits`].
    pub fn getrandbits_u64(&mut self, bits: u32) -> Result<u64, RandomError> {
        if bits > 64 {
            return Err(RandomError::BitsTooWide);
        }
        Ok(self
            .getrandbits(bits as usize)
            .to_u64()
            .expect("at most 64 generated bits always fit in u64"))
    }

    /// CPython's `_randbelow_with_getrandbits(n)` for a positive native bound.
    pub fn randbelow(&mut self, bound: usize) -> Result<usize, RandomError> {
        if bound == 0 {
            return Err(RandomError::EmptyRange);
        }
        let bits = (usize::BITS - bound.leading_zeros()) as usize;
        loop {
            let candidate = self
                .getrandbits(bits)
                .to_u64()
                .expect("a usize-width sample always fits in u64");
            if candidate < bound as u64 {
                return Ok(candidate as usize);
            }
        }
    }

    /// The one-argument positive-stop form of Python's `randrange`.
    pub fn randrange(&mut self, stop: usize) -> Result<usize, RandomError> {
        self.randbelow(stop)
    }

    /// Select an index exactly as Python's `choice` does.
    pub fn choice_index(&mut self, length: usize) -> Result<usize, RandomError> {
        if length == 0 {
            return Err(RandomError::EmptySequence);
        }
        self.randbelow(length)
    }

    /// Select a borrowed element exactly as Python's `choice` does.
    pub fn choice<'a, T>(&mut self, values: &'a [T]) -> Result<&'a T, RandomError> {
        let index = self.choice_index(values.len())?;
        Ok(&values[index])
    }

    /// In-place Fisher-Yates shuffle using Python's `_randbelow` draw sequence.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let swap_with = self
                .randbelow(index + 1)
                .expect("a shuffle bound is always positive");
            values.swap(index, swap_with);
        }
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..N {
            let previous = self.state[index - 1];
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut state_index, mut key_index) = (1_usize, 0_usize);
        for _ in 0..N.max(key.len()) {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[key_index])
            .wrapping_add(key_index as u32);
            state_index += 1;
            key_index += 1;
            if state_index >= N {
                self.state[0] = self.state[N - 1];
                state_index = 1;
            }
            if key_index >= key.len() {
                key_index = 0;
            }
        }
        for _ in 1..N {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(state_index as u32);
            state_index += 1;
            if state_index >= N {
                self.state[0] = self.state[N - 1];
                state_index = 1;
            }
        }
        self.state[0] = UPPER_MASK;
        self.index = N;
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;

        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn twist(&mut self) {
        for index in 0..(N - M) {
            let value = (self.state[index] & UPPER_MASK) | (self.state[index + 1] & LOWER_MASK);
            self.state[index] =
                self.state[index + M] ^ (value >> 1) ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        }
        for index in (N - M)..(N - 1) {
            let value = (self.state[index] & UPPER_MASK) | (self.state[index + 1] & LOWER_MASK);
            self.state[index] = self.state[index - (N - M)]
                ^ (value >> 1)
                ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        }
        let value = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[N - 1] =
            self.state[M - 1] ^ (value >> 1) ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        self.index = 0;
    }
}

fn multiply_add_words(words: &mut Vec<u32>, multiplier: u32, addend: u32) {
    let mut carry = u64::from(addend);
    for word in words.iter_mut() {
        let value = u64::from(*word) * u64::from(multiplier) + carry;
        *word = value as u32;
        carry = value >> 32;
    }
    if carry != 0 {
        words.push(carry as u32);
    }
}

fn trim_high_zero_words(words: &mut Vec<u32>) {
    while words.last() == Some(&0) {
        words.pop();
    }
}
