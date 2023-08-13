use crate::ofdm::ofdm_preamble_encode;
use clap::{builder::RangedU64ValueParser, Args, Parser, Subcommand};
use std::ops::Range;
use stft::fft::FourierFloat;

/// The spec for a transmission.
// TODO Include things like realtime_vs_file
#[derive(Clone, Debug, Subcommand)]
pub enum TransmissionSpec {
    Ofdm(OfdmSpec),
    Fdm(FdmSpec),
}

#[derive(Parser, Clone, Debug, Default)]
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

#[derive(Args, Clone, Debug)]
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

    /// Number simultaneous bytes per symbol.
    #[arg(short = 'b', long, default_value_t = 8, value_parser = RangedU64ValueParser::<usize>::new().range(1..=u64::MAX))]
    pub simultaneous_bytes: usize,
}

impl OfdmSpec {
    /// Preamble of each frame.
    pub fn preamble<T: FourierFloat>(&self) -> impl Iterator<Item = T> + Clone + core::fmt::Debug {
        ofdm_preamble_encode(self)
    }

    /// Bins that transmit data.
    // TODO take subcarrier type into consideration. This will allocate too many bins if using a multi-bit bin encoding ex qpsk.
    pub fn active_bins(&self) -> Range<usize> {
        self.first_bin..(self.first_bin + 8 * self.simultaneous_bytes)
    }

    /// The bits sent in a symbol
    pub fn bits_per_symbol(&self) -> usize {
        8 * self.simultaneous_bytes
    }

    /// Length of the length field.
    pub fn len_field_len(&self) -> usize {
        let length_field_bits: usize = usize::BITS.try_into().unwrap();
        self.symbol_len()
            * (length_field_bits / self.bits_per_symbol()
                + usize::from(length_field_bits % self.bits_per_symbol() != 0))
    }

    /// Length of symbol + cyclic prefix.
    pub fn symbol_len(&self) -> usize {
        self.time_symbol_len + self.cyclic_prefix_len
    }

    /// Length of frame.
    pub fn frame_len(&self) -> usize {
        self.preamble::<f32>().count() // premable
        + self.len_field_len() // length field
        + self.data_symbols * self.symbol_len() // data symbols
    }
}

impl Default for OfdmSpec {
    fn default() -> Self {
        use clap::{Command, FromArgMatches};
        let c = Command::new("default");
        let c = <Self as Args>::augment_args(c);
        let m = c.get_matches_from([""].iter());
        <OfdmSpec as FromArgMatches>::from_arg_matches(&m).expect("hardcoded values work")
    }
}
