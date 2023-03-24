use bitvec::prelude::{BitArray, LocalBits, Lsb0};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use log::{error, trace};
use std::f32::consts::PI;
use std::iter::{self, Chain, Iterator};
use std::marker::Send;

/// Writes the data given by the closure onto the output stream.
fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

/// Main logic for sending sounds through the speaker.
fn run<T>(
    device: &Device,
    config: &StreamConfig,
    mut audio_data: impl Iterator<Item = f32> + Send + Sync + 'static,
) -> anyhow::Result<MustUse<Stream>>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;

    let mut next_value = move || match audio_data.next() {
        Some(x) => x,
        None => Sample::EQUILIBRIUM,
    };

    let err_fn = |err| error!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value);
        },
        err_fn,
        None,
    )?;
    trace!("Playing stream");
    stream.play()?;
    Ok(MustUse(stream))
}

/// Selects the correct data type depending on supported format and then runs.
fn play_stream(
    device: &Device,
    frequency: impl Iterator<Item = f32> + Sync + Send + 'static,
) -> anyhow::Result<MustUse<Stream>> {
    let config = device.default_output_config().unwrap();
    trace!("Default output config: {:?}", config);

    match config.sample_format() {
        SampleFormat::I8 => run::<i8>(device, &config.into(), frequency),
        SampleFormat::I16 => run::<i16>(device, &config.into(), frequency),
        SampleFormat::I32 => run::<i32>(device, &config.into(), frequency),
        SampleFormat::I64 => run::<i64>(device, &config.into(), frequency),
        SampleFormat::U8 => run::<u8>(device, &config.into(), frequency),
        SampleFormat::U16 => run::<u16>(device, &config.into(), frequency),
        SampleFormat::U32 => run::<u32>(device, &config.into(), frequency),
        SampleFormat::U64 => run::<u64>(device, &config.into(), frequency),
        SampleFormat::F32 => run::<f32>(device, &config.into(), frequency),
        SampleFormat::F64 => run::<f64>(device, &config.into(), frequency),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
}

/// Annotates type as must use.
/// Useful when the inner type (eg. of a result) must be used even if the outer type is used (eg. result contains important data that must bind).
#[must_use = "The inner type must be used."]
pub struct MustUse<T>(T);

impl<T> From<T> for MustUse<T> {
    fn from(v: T) -> Self {
        Self(v)
    }
}

/// Returns a stream that plays sound samples until the returned stream is dropped. If the iterator runs out it plays silence (equilibrium samples).
pub fn play_freq(
    frequency: impl Iterator<Item = f32> + Send + Sync + 'static,
) -> anyhow::Result<MustUse<Stream>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    trace!("Output device: {}", device.name().unwrap());

    play_stream(&device, frequency)
}

/// Coverts byte to iterator of bool.
fn byte_to_bit(byte: u8) -> bitvec::array::IntoIter<u8, LocalBits> {
    BitArray::<u8, Lsb0>::from(byte).into_iter()
}

/// Return type of `bytes_to_bits`. Basically `impl Iterator<Item = bool>`.
type BitIter<T> = iter::FlatMap<
    T,
    bitvec::array::IntoIter<u8, Lsb0>,
    fn(u8) -> bitvec::array::IntoIter<u8, Lsb0>,
>;

/// Converts an iterator of bytes into a 8x longer iterator of bits.
/// The returned type is impl Iterator<Item = bool> but keeps other traits (ex. `Clone`) the original iterator has.
pub fn bytes_to_bits<T: Iterator<Item = u8>>(bytes: T) -> BitIter<T> {
    bytes.into_iter().flat_map(byte_to_bit)
}

/// All parameters needed for amplitude modulation.
#[derive(Debug, Default, Clone)]
pub struct AmplitudeModulationConfig {
    channel_width: f32,           // Frequency channel width.
    parallel_channels_num: usize, // Simultaneous frequency channels for transmission.
    sample_rate: f32,             // Samples per second.
    start_freq: f32,              // Lower bound of lowest frequency channel.
    symbol_length: usize,         // The number of samples per symbol.
}

/// The iterator for the type of the encoded amplitude modulation. It maintains traits (ex. `Clone`) of the original iterator.
/// It prepends and appends a guard symbol to the iterator.
type IterWithGuard<T> =
    Chain<Chain<iter::Take<iter::Repeat<bool>>, T>, iter::Take<iter::Repeat<bool>>>;

impl AmplitudeModulationConfig {
    /// Basic constructor
    /// # Panics
    /// - Max frequency required for encoding must be less than Nyquist Frequency.
    /// The max frequency is determined by simultaneous channels and starting frequency.
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
        AmplitudeModulationConfig {
            channel_width,
            parallel_channels_num,
            sample_rate,
            start_freq,
            symbol_length,
        }
    }

    /// Creates new iterator over encoded values.
    pub fn encode<T: Iterator<Item = bool>>(
        &self,
        bits: T,
    ) -> AmplitudeModulationSignal<IterWithGuard<T>> {
        let guard = iter::repeat(true).take(self.parallel_channels_num);
        let guard_end = guard.clone();
        let bits = guard.chain(bits).chain(guard_end);
        AmplitudeModulationSignal {
            bits,
            symbol_frequencies: Vec::new(),
            symbol_idx: self.symbol_length,
            config: self,
        }
    }
}

/// Each item in the iterator is the next audio sample for transmitting the given bits.
/// Iterator generates the samples for an amplitude modulated transmission from bits.
#[derive(Debug, Clone)]
pub struct AmplitudeModulationSignal<'a, T: Iterator<Item = bool>> {
    bits: T,                      // The bits to transmit.
    symbol_frequencies: Vec<f32>, // The frequencies to transmit for the symbol.
    config: &'a AmplitudeModulationConfig,
    symbol_idx: usize, // The idx for the next sample.
}

impl<T: Iterator<Item = bool>> AmplitudeModulationSignal<'_, T> {}

impl<T: Iterator<Item = bool>> Iterator for AmplitudeModulationSignal<'_, T> {
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

#[cfg(test)]
mod tests {
    use super::AmplitudeModulationConfig;

    #[test]
    fn size_one_bit() {
        let bits = [true].into_iter();
        let channel_width = 1.0;
        let parallel_channels_num = 1;
        let sample_rate = 4.0;
        let start_freq = 1.0;
        let symbol_length = 1;
        let config = AmplitudeModulationConfig::new(
            channel_width,
            parallel_channels_num,
            sample_rate,
            start_freq,
            symbol_length,
        );
        let iter = config.encode(bits);
        // If there is one bit it should be transmitted for one symbol length + 2 guard symbols;
        let signal_length = symbol_length + 2 * symbol_length;
        assert_eq!(signal_length, iter.clone().count());
        assert_eq!(iter.clone().count(), iter.clone().size_hint().0);
        assert_eq!(iter.clone().count(), iter.clone().size_hint().1.unwrap());

        // Size hint should stay correct
        let mut iter = iter.enumerate();
        while let Some((i, _)) = iter.next() {
            assert_eq!(
                signal_length - 1 - i,
                iter.clone().count(),
                "Length failed on iteration {i}"
            );
            assert_eq!(
                iter.clone().count(),
                iter.clone().size_hint().0,
                "Lower bound failed on iteration {i}"
            );
            assert_eq!(
                iter.clone().count(),
                iter.clone().size_hint().1.unwrap(),
                "Upper bound failed on iteration {i}"
            );
        }
    }

    #[test]
    fn size_multi_bit() {
        let bits = [false, true].into_iter();
        let channel_width = 1.0;
        let parallel_channels_num = 2;
        let sample_rate = 6.0;
        let start_freq = 1.0;
        let symbol_length = 2;
        let config = AmplitudeModulationConfig::new(
            channel_width,
            parallel_channels_num,
            sample_rate,
            start_freq,
            symbol_length,
        );
        let iter = config.encode(bits);

        // If there are two bits in parallel should be transmitted for one symbol length + 2 guard symbol;
        let signal_length = symbol_length + 2 * symbol_length;
        assert_eq!(signal_length, iter.clone().count());
        assert_eq!(iter.clone().count(), iter.clone().size_hint().0);
        assert_eq!(iter.clone().count(), iter.clone().size_hint().1.unwrap());

        // Size hint should stay correct
        let mut iter = iter.enumerate();
        while let Some((i, _)) = iter.next() {
            assert_eq!(
                signal_length - 1 - i,
                iter.clone().count(),
                "Length failed on iteration {i}"
            );
            assert_eq!(
                iter.clone().count(),
                iter.clone().size_hint().0,
                "Lower bound failed on iteration {i}"
            );
            assert_eq!(
                iter.clone().count(),
                iter.clone().size_hint().1.unwrap(),
                "Upper bound failed on iteration {i}"
            );
        }
    }
}
