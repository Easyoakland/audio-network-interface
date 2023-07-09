use clap::{builder::RangedU64ValueParser, Args, Subcommand};

/// The spec for a transmission.
// TODO Include things like realtime_vs_file
#[derive(Clone, Debug, Subcommand)]
pub enum TransmissionSpec {
    Ofdm(OfdmSpec),
    Fdm(FdmSpec),
}

#[derive(Args, Clone, Debug, Default)]
/// Basic frequency division multiplexing.
pub struct FdmSpec {
    /// The time per symbol in milliseconds.
    #[arg(short = 't', long, default_value_t = 100)]
    pub symbol_time: u64,

    /// The width allocated to a bit in hertz. Total bandwith for a byte is 8 * width.
    #[arg(short = 'w', long, default_value_t = 20.0)]
    pub bit_width: f32,

    /// The starting frequency of the transmission in hertz.
    #[arg(short, long, default_value_t = 1000.0)]
    pub start_freq: f32,

    /// The number of parallel channels to use for transmission.
    #[arg(short, long, default_value_t = 8)]
    pub parallel_channels: usize,
}

#[derive(Args, Clone, Debug, Default)]
/// Orthogonal frequency division multiplexing. Commonly used in digital communication such as in Wifi and 4G technology.
pub struct OfdmSpec {
    /// Seed used for pseudorandom generation.
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,

    /// Number of repetitions in the preamble short training sequence.
    #[arg(short = 'r', long, default_value_t = 10)]
    pub short_training_repetitions: usize,

    /// Length of symbol in time samples.
    #[arg(short, long, default_value_t = 4800)]
    pub time_symbol_len: usize,

    /// Length of cyclic prefix in time samples.
    #[arg(short, long, default_value_t = 480)]
    pub cyclic_prefix_len: usize,

    /// The autocorrelation threshold to use when finding detecting frames.
    /// Range should be from 1.0 (exactly the same signal) to 0.0 (uncorrelated).
    #[arg(short = 'T', long, default_value_t = 0.125)]
    pub cross_correlation_threshold: f32,

    /// The number of data symbols in a frame.
    #[arg(short, long, default_value_t = 32)]
    pub data_symbols: usize,

    /// First subcarrier frequency bin index.
    /// Can't be 0 because dc bin can't tranmit data.
    #[arg(short, long, default_value_t = 20, value_parser = RangedU64ValueParser::<usize>::new().range(1..=u64::MAX))]
    pub first_bin: usize,
}
