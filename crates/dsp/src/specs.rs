use crate::ofdm::ofdm_preamble_encode;
use clap::{Args, Parser, Subcommand};
use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::RangeInclusive,
    str::FromStr,
};
use stft::{fft::FourierFloat, frequency_samples_to_time};

#[derive(Debug)]
pub enum ParseRangeError<E> {
    Inner(E),
    Sep,
}
impl<E: Display> Display for ParseRangeError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseRangeError::Inner(e) => e.fmt(f),
            ParseRangeError::Sep => write!(f, "no valid seperator found"),
        }
    }
}

impl<E: Display + Debug> Error for ParseRangeError<E> {}

impl<E> From<E> for ParseRangeError<E> {
    fn from(value: E) -> Self {
        Self::Inner(value)
    }
}

#[derive(Debug, Clone)]
pub struct ParsedRangeInclusive<T>(pub RangeInclusive<T>);

impl<T: Display> Display for ParsedRangeInclusive<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.0.start(), self.0.end())
    }
}

impl<T: FromStr> FromStr for ParsedRangeInclusive<T> {
    type Err = ParseRangeError<T::Err>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((start, end)) = s.split_once('-') {
            Ok(Self((start.parse::<T>()?)..=(end.parse::<T>()?)))
        } else {
            Err(ParseRangeError::Sep)
        }
    }
}

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

    /*
    /// Raw length of symbol in time samples. Proportional to frequency bins.
    #[arg(short, long, default_value_t = 4800)]
    pub time_symbol_len: usize,
    */
    /// Number of frequency bins to divide the frequency spectrum into. More bins allows finer frequencies and more bits per symbol, but requires longer time per symbol.
    #[arg(short = 'n', long, default_value_t = 2401, value_parser = clap::builder::RangedU64ValueParser:: <usize> ::new().range(1..=u64::MAX))]
    pub bin_num: usize,

    /// Frequency bins used to transmit. Bits per symbol depends on this and subcarrier modulation scheme.
    /// Start can't be 0 because dc bin can't transmit data.
    #[arg(short = 'b', long, default_value_t = ParsedRangeInclusive(20..=83))]
    pub active_bins: ParsedRangeInclusive<usize>,

    /// Length of cyclic prefix in time samples.
    #[arg(short, long, default_value_t = 480)]
    pub cyclic_prefix_len: usize,

    /// The autocorrelation threshold to use when detecting frames.
    /// Range should be from 1.0 (exactly the same signal) to 0.0 (uncorrelated).
    #[arg(short = 'T', long, default_value_t = 0.125)]
    pub cross_correlation_threshold: f32,

    /// The number of data symbols in a frame.
    #[arg(short, long, default_value_t = 32)]
    pub data_symbols: usize,
}

impl OfdmSpec {
    /// Preamble of each frame.
    pub fn preamble<T: FourierFloat>(&self) -> impl Iterator<Item = T> + Clone + core::fmt::Debug {
        ofdm_preamble_encode(self)
    }

    /// Number of frequency bins
    pub fn bin_num(&self) -> usize {
        // time_samples_to_frequency(self.time_symbol_len)
        self.bin_num
    }

    /// Number of samples in time
    pub fn time_symbol_len(&self) -> usize {
        frequency_samples_to_time(self.bin_num)
    }

    /// Bins that transmit data.
    pub fn active_bins(&self) -> impl Iterator<Item = usize> + Clone + Debug {
        self.active_bins.0.clone()
    }

    /// The bits sent in a symbol
    // TODO take subcarrier type into consideration. This currently assumes 1bit/bin.
    pub fn bits_per_symbol(&self) -> usize {
        self.active_bins().map(|_| 1).sum()
    }

    /// Length of symbol + cyclic prefix.
    pub fn symbol_samples_len(&self) -> usize {
        self.time_symbol_len() + self.cyclic_prefix_len
    }

    /// Length of the length field.
    pub fn len_field_samples_len(&self) -> usize {
        let length_field_bits: usize = usize::BITS.try_into().unwrap();
        self.symbol_samples_len()
            * (length_field_bits / self.bits_per_symbol()
                + usize::from(length_field_bits % self.bits_per_symbol() != 0))
    }

    /// Length of samples transmitting data per frame.
    pub fn data_samples_len(&self) -> usize {
        self.data_symbols * self.symbol_samples_len()
    }

    /// Length of frame.
    pub fn frame_len(&self) -> usize {
        self.preamble::<f32>().count() // premable
        + self.len_field_samples_len() // length field
        + self.data_samples_len() // data symbols
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
