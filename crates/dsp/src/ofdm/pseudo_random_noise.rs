use num_complex::Complex;
use rand::seq::IteratorRandom;
use rand_core::SeedableRng;
use rand_pcg::Pcg32;
use std::cmp::PartialOrd;

/// Pseudo random noise values generated for short preamble.
/// Taken from IEEE 802.11 standard Table L-2 and L-4.
pub const SHORT_PRN: [Complex<f32>; 64] = [
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: -1.472,
        im: -1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: -1.472,
        im: -1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: -1.472,
        im: -1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: -1.472,
        im: -1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: -1.472,
        im: -1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex {
        re: 1.472,
        im: 1.472,
    },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
];
/// Pseudo random noise values generated for long preamble.
/// Taken from IEEE 802.11 standard Table L-2 and L-4.
pub const LONG_PRN: [Complex<f32>; 64] = [
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: -1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 1.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
    Complex {
        re: 0.000,
        im: 0.000,
    },
];

/// Generate pseudorandom sequence of numbers.
/// Deterministic by seed.
pub fn gen_pseudonoise_sequence<T>(
    seed: u64,
    n: usize,
    iterator: impl Iterator<Item = T> + Clone,
) -> impl Iterator<Item = T>
where
    T: rand::distributions::uniform::SampleUniform + PartialOrd + Clone + 'static,
{
    let mut rng = Pcg32::seed_from_u64(seed);
    (0..n).flat_map(move |_| iterator.clone().choose_stable(&mut rng))
}
