pub const MAX_THRESHOLD_PERCENT: usize = 90;
use bitvec::{bitvec, prelude::Lsb0, vec::BitVec};
use iterator_adapters::IteratorAdapter;
use log::trace;
use ordered_float::OrderedFloat;
use std::{
    f32::consts::PI,
    io::Read,
    iter::{self, Chain},
};
use stft::Stft;

/// All parameters needed for ook fdm modulation.
#[derive(Debug, Default, Clone)]
pub struct OokFdmConfig {
    /// Frequency channel width.
    channel_width: f32,
    /// Simultaneous frequency channels for transmission.
    parallel_channels_num: usize,
    /// Samples per second.
    sample_rate: f32,
    /// Lower bound of lowest frequency channel.
    start_freq: f32,
    /// The number of samples per symbol.
    symbol_length: usize,
}

/// The iterator for the type of the encoded ook fdm modulation. It maintains traits (ex. `Clone`) of the original iterator.
/// It prepends and appends a guard symbol to the iterator.
type IterWithGuard<T> =
    Chain<Chain<iter::Take<iter::Repeat<bool>>, T>, iter::Take<iter::Repeat<bool>>>;

impl OokFdmConfig {
    /// Basic constructor
    /// # Panics
    /// - Max frequency required for encoding must be less than Nyquist Frequency.
    /// The max frequency is determined by simultaneous channels and starting frequency.
    #[must_use]
    pub fn new(
        channel_width: f32,
        parallel_channels_num: usize,
        sample_rate: f32,
        start_freq: f32,
        symbol_length: usize,
    ) -> Self {
        let max_freq = start_freq + parallel_channels_num as f32 * channel_width;
        assert!(
            sample_rate/2.0 >= max_freq,
            "Sample rate not high enough. For encoding specified parameters sample rate must be at least 2x{max_freq}. Try decreasing parallel channels and starting frequency."
        );
        OokFdmConfig {
            channel_width,
            parallel_channels_num,
            sample_rate,
            start_freq,
            symbol_length,
        }
    }

    /// Creates new iterator over encoded values.
    pub fn encode<T: Iterator<Item = bool>>(&self, bits: T) -> OokFdmSignal<IterWithGuard<T>> {
        let guard = iter::repeat(true).take(self.parallel_channels_num);
        let guard_end = guard.clone();
        let bits = guard.chain(bits).chain(guard_end);
        OokFdmSignal {
            bits,
            symbol_frequencies: Vec::new(),
            symbol_idx: self.symbol_length,
            config: self,
        }
    }
}

/// Each item in the iterator is the next audio sample for transmitting the given bits.
/// Iterator generates the samples for an ook fdm modulated transmission from bits.
#[derive(Debug, Clone)]
pub struct OokFdmSignal<'a, T: Iterator<Item = bool>> {
    bits: T,                      // The bits to transmit.
    symbol_frequencies: Vec<f32>, // The frequencies to transmit for the symbol.
    config: &'a OokFdmConfig,
    symbol_idx: usize, // The idx for the next sample.
}

impl<T: Iterator<Item = bool>> Iterator for OokFdmSignal<'_, T> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        {
            // If previous symbol has already been generated for symbol length start the next symbol.
            if self.symbol_idx >= self.config.symbol_length {
                // Reset internal trackers for iterator.
                self.symbol_frequencies.clear();
                self.symbol_idx = 0;

                let mut chunk = Vec::with_capacity(self.config.parallel_channels_num);
                let mut i = 0;
                // Get the next chunk of bits.
                while i < self.config.parallel_channels_num {
                    chunk.push(match self.bits.next() {
                        Some(x) => x,
                        None => {
                            if i == 0 {
                                // If the bit iterator is already exhausted then there are no more samples to generate.
                                return None;
                            } else {
                                false
                            }
                        }
                    });
                    i += 1;
                }
                // For each bit in the chunk generate the corresponding sinusoid for that frequency.
                for (i, bit) in chunk.into_iter().enumerate() {
                    match bit {
                        true => self.symbol_frequencies.push(
                            self.config.start_freq + (1 + i) as f32 * self.config.channel_width,
                        ),
                        false => (),
                    }
                }
            }
            // Generate next sample.
            let t = self.symbol_idx as f32 / self.config.sample_rate;
            // Add all frequencies together and normalize by number of channels so each channel's amplitude is consistent.
            let sum: f32 = self
                .symbol_frequencies
                .iter()
                .map(|f| (2.0 * PI * f * t).sin())
                .sum();
            // Increment to next position of symbol.
            self.symbol_idx += 1;
            // Produce sinusoid of maximum amplitude.
            Some(sum / self.config.parallel_channels_num as f32)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.bits.size_hint().0 / self.config.parallel_channels_num * self.config.symbol_length
                + self.config.symbol_length
                - self.symbol_idx,
            self.bits
                .size_hint()
                .1
                .map(|x| x / self.config.parallel_channels_num)
                .map(|x| x * self.config.symbol_length)
                .map(|x| x + self.config.symbol_length - self.symbol_idx),
        )
    }
}

/// Decodes an ook fdm modulated audio signal.
#[derive(Debug, Default, Clone)]
pub struct OokFdmDecoder {
    pub frequency_channels: Vec<f32>,
    pub sample_rate: f32,
}

impl OokFdmDecoder {
    /// Decodes frequency analysis from [`stft::SpecCompute::stft()`] into data bits.
    /// Sensitivity is from 0 (most sensitive) to 1 (least sensitive).
    pub fn decode_ook_fdm(&self, stft: &Stft, sensitivity: f64) -> Vec<u8> {
        assert!(
            (0.0..=1.0).contains(&sensitivity),
            "Invalid sensitivity. Must be between 0.0 - 1.0"
        );
        let frequency_channel_cnt = self.frequency_channels.len();
        let mut decoded: BitVec<u8, Lsb0> =
            bitvec![u8, Lsb0; 0; frequency_channel_cnt * stft.window_cnt()];
        // Ex. f 1 1 1
        //       1 0 1
        //       t
        // reads 1,0,1 of first frequency with f_idx = 0, v_idx = 0,1,2
        // so indices in self correspond to 0,2,4
        // next reads 1,1,1 of second frequency with f_idx = 1, v_idx = 0,1,2
        // so indices in self correspond to 1,3,5
        for (f_idx, frequency) in self.frequency_channels.iter().enumerate() {
            let bin = stft.get_bin(*frequency, self.sample_rate).unwrap();
            let threshold = bin
                .iter()
                .copied()
                .map(OrderedFloat)
                .kth_order(MAX_THRESHOLD_PERCENT)
                .0
                / 2.0;
            let threshold = threshold * sensitivity;
            trace!("Threshold for frequency {frequency} Hz is {threshold}");
            for (v_idx, value) in bin.iter().enumerate() {
                if value > &threshold {
                    decoded.set(f_idx + v_idx * frequency_channel_cnt, true);
                }
            }
        }

        // When window first full of active bits the packet is started the bit after that window.
        let start_idx = decoded
            .windows(frequency_channel_cnt)
            .position(|x| x.iter().filter(|x| **x).count() == frequency_channel_cnt)
            .expect("Can't find start.")
            + frequency_channel_cnt;
        // When window is first full of active bits in reverse the packet ends at the end of that window.
        let end_idx = decoded
            .windows(frequency_channel_cnt)
            .rev()
            // When window is full of active bits the packet is started.
            .position(|x| x.iter().filter(|x| **x).count() == frequency_channel_cnt)
            .expect("Can't find end.")
            + frequency_channel_cnt;
        trace!(
            "Start at {start_idx}. End at {}",
            decoded.len() - 1 - end_idx
        );
        assert_ne!(
            start_idx,
            decoded.len() - 1 - end_idx,
            "Start idx is same as end idx"
        );
        assert!(start_idx < decoded.len() - 1 - end_idx, "End before start.");

        // Remove stuff before and after guard symbol.
        let decoded: BitVec<u8, Lsb0> = decoded
            .into_iter()
            .skip(start_idx)
            .rev()
            .skip(end_idx)
            .rev()
            .collect();

        // Convert bit vector to byte vector.
        let bytes: Result<Vec<u8>, _> = decoded.bytes().collect();
        bytes.expect("Failed to convert decoded to bytes.")
    }
}
