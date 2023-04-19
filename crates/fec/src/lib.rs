/// <https://en.wikipedia.org/wiki/Hamming_code>
/// <https://en.wikipedia.org/wiki/Hamming(7,4)>
pub mod hamming;

/// Methods of adding parity checks into the data.
pub mod parity;

/// Methods for utilizing reed solomon erasure encoding.
pub mod reed_solomon;

/// Methods for utilizing convolutional encoding.
mod viterbi {}

/// Traits for generic processing
pub mod traits;
