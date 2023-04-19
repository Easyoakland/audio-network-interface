use self::pseudo_random_noise::gen_pseudonoise_sequence;
use crate::{
    correlation::{cross_correlation_timing_metric, cross_correlation_timing_metric_single_value},
    specs::OfdmSpec,
};
use bitvec::vec::BitVec;
use iterator_adapters::IteratorAdapter;
use num_complex::Complex;
use num_traits::{One, Zero};
use std::{
    fmt::Debug,
    iter::{self, Iterator, Sum},
    vec,
};
use stft::{
    fft::{scaled_real_fft, scaled_real_ifft, FourierFloat},
    frequency_samples_to_time, time_samples_to_frequency,
};

/// Pseudo random noise used in ofdm preamble symbols.
pub mod pseudo_random_noise;

/// Function that takes a fixed number of inputs and outputs something.
#[derive(Clone, Copy, Debug)]
pub enum SubcarrierEncoder<I, O> {
    T0(fn([I; 0]) -> O),
    T1(fn([I; 1]) -> O),
    T2(fn([I; 2]) -> O),
    T3(fn([I; 3]) -> O),
    T4(fn([I; 4]) -> O),
    T5(fn([I; 5]) -> O),
    T6(fn([I; 6]) -> O),
}

impl<I, O> From<SubcarrierEncoder<I, O>> for usize {
    fn from(value: SubcarrierEncoder<I, O>) -> Self {
        match value {
            SubcarrierEncoder::T0(_) => 0,
            SubcarrierEncoder::T1(_) => 1,
            SubcarrierEncoder::T2(_) => 2,
            SubcarrierEncoder::T3(_) => 3,
            SubcarrierEncoder::T4(_) => 4,
            SubcarrierEncoder::T5(_) => 5,
            SubcarrierEncoder::T6(_) => 6,
        }
    }
}

/// Function that takes a fixed number of inputs and outputs something.
#[derive(Clone, Copy)]
pub enum SubcarrierDecoder<'a, T> {
    Data(fn(Complex<T>) -> Vec<bool>), // Converts symbol to corresponding data.
    Pilot(&'a dyn Fn(Complex<T>) -> Complex<T>), // Converts pilot to scale factor.
}

impl<'a, T> Debug for SubcarrierDecoder<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(arg0) => f.debug_tuple("Data").field(arg0).finish(),
            Self::Pilot(_) => f.debug_tuple("Pilot").finish(),
        }
    }
}

/// Ofdm data encoding.
#[derive(Clone, Debug)]
pub struct OfdmDataEncoder<'a, I: Iterator<Item = bool>, T: FourierFloat> {
    bits: std::iter::Chain<I, std::iter::Once<bool>>, // The bits to transmit plus a bit to indicate the end of data.
    channels: &'a [SubcarrierEncoder<bool, Complex<T>>], // Each channel has a function that converts some finite number of bits into a complex value.
    scratch_space: Vec<Complex<T>>, // Load the next value for each channel into this space instead of reallocating a vector.
    cyclic_prefix_length: usize, // Number of samples to add from the end of the time signal onto the beginning.
}

impl<'a, I, T> OfdmDataEncoder<'a, I, T>
where
    I: Iterator<Item = bool>,
    T: FourierFloat,
{
    /// Creates a new OFDM modulation where each subcarrier's values are determined by the subcarrier's function.
    /// # Arguments
    /// - `bits`: The underlying bit source that is being modulated.
    /// - `subcarrier_functions`: The function to use at a given sub carrier.
    /// - `cyclic_prefix_length`: The number of time samples to repeat from the end of the sample at the beginning as a cyclic prefix. Must be <= time domain symbol sample length.
    pub fn new(
        bits: I,
        subcarrier_functions: &'a [SubcarrierEncoder<bool, Complex<T>>],
        cyclic_prefix_length: usize,
    ) -> Self {
        assert!(
            cyclic_prefix_length <= frequency_samples_to_time(subcarrier_functions.len()),
            "Cyclic prefix can't be longer than the actual symbol."
        );
        OfdmDataEncoder {
            scratch_space: vec![Complex::zero(); subcarrier_functions.len()],
            channels: subcarrier_functions,
            bits: bits.chain(iter::once(true)),
            cyclic_prefix_length,
        }
    }

    /// The size hint for the number of remaining bits (not symbols).
    pub fn bits_left(&self) -> (usize, Option<usize>) {
        self.bits.size_hint()
    }

    /// The number of positive frequency domain samples per symbol.
    pub fn freq_len(&self) -> usize {
        self.channels.len()
    }

    /// The time domain sample length per symbol without cyclic prefix.
    pub fn time_len(&self) -> usize {
        frequency_samples_to_time(self.channels.len())
    }

    /// The length of the cyclic prefix.
    pub fn cyclic_len(&self) -> usize {
        self.cyclic_prefix_length
    }

    /// The length of the time domain symbol for both cyclic prefix and data.
    pub fn symbol_len(&self) -> usize {
        self.time_len() + self.cyclic_len()
    }
}

impl<'a, I, T> Iterator for OfdmDataEncoder<'a, I, T>
where
    I: Iterator<Item = bool>,
    T: FourierFloat,
{
    type Item = Vec<T>;

    /// Use the function for each subcarrier to determine its complex value to transmit.
    /// Use these complex values for each channel in an ifft to determine the next symbol's time samples.
    /// Will pull bits from the bit source to give to each function the amount of bits it needs in order.
    fn next(&mut self) -> Option<Self::Item> {
        /// TODO fix this mess.
        /// I can't determine a better way to do this but this looks bad.
        /// It applies the subcarrier modulation regardless of how long the input array is.
        fn logic<const N: usize, I: Iterator<Item = bool>, T>(
            bits: &mut I,
            f: fn([bool; N]) -> Complex<T>,
            scratch: &mut Complex<T>,
        ) -> usize {
            // Pull the necessary bits from source into an array.
            let mut bits_for_channel = [Default::default(); N];

            // TODO: Handle bits of not integer multiple of subcarriers number.
            // If runs out of data leaves the default value.
            let mut i = 0;
            while i < N {
                bits_for_channel[i] = match bits.next() {
                    Some(x) => x,
                    None => break,
                };
                i += 1;
            }

            // Perform next sample computation.
            *scratch = f(bits_for_channel);

            i
        }
        // Pull next values into scratch.
        let mut first_subcarrier = true;
        for (f, scratch) in self.channels.iter().zip(self.scratch_space.iter_mut()) {
            #[rustfmt::skip]
            match f {
                SubcarrierEncoder::T0(f) => { logic(&mut self.bits, *f, scratch); },
                SubcarrierEncoder::T1(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
                SubcarrierEncoder::T2(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
                SubcarrierEncoder::T3(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
                SubcarrierEncoder::T4(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
                SubcarrierEncoder::T5(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
                SubcarrierEncoder::T6(f) => if logic(&mut self.bits, *f, scratch) == 0 && first_subcarrier { return None } else { first_subcarrier = false },
            };
        }

        // Generate time series for next OFDM symbol.
        let raw_time_symbol = scaled_real_ifft(&mut self.scratch_space);

        // Prefix cyclic prefix.
        let cyclic_prefix = &raw_time_symbol[raw_time_symbol.len() - self.cyclic_prefix_length..];
        let symbol_with_cyclic_prefix = {
            let mut out = Vec::with_capacity(raw_time_symbol.len() + cyclic_prefix.len());
            out.extend_from_slice(cyclic_prefix);
            out.extend_from_slice(&raw_time_symbol);
            out
        };
        Some(symbol_with_cyclic_prefix)
    }
}

/// All parameters needed for ofdm demodulation.
/// Implements iteration over decoded values. The final partial symbol is not filtered directly by the iterator.
/// To get the actual value minus the padding on the final symbol use [`Self::decode`].
#[derive(Clone, Debug)]
pub struct OfdmDataDecoder<'a, I: Iterator<Item = T>, T: FourierFloat, const CHANNELS_NUM: usize> {
    samples: I,                                                    // The bits to transmit.
    subcarrier_decoders: [SubcarrierDecoder<'a, T>; CHANNELS_NUM], // Each channel has a function that converts some complex number into some number of bits.
    scratch_space: Vec<T>, // Load the next value for each channel into this space instead of reallocating a vector.
    cyclic_prefix_length: usize, // Number of samples to add from the end of the time signal onto the beginning.
    gain_factors: [Complex<T>; CHANNELS_NUM], // The factor to adjust each subcarrier.
}

impl<'a, I, T, const CHANNELS_NUM: usize> OfdmDataDecoder<'a, I, T, CHANNELS_NUM>
where
    I: Iterator<Item = T>,
    T: FourierFloat,
{
    /// Creates a new OFDM demodulation where each subcarrier's values are determined by the subcarrier's function.
    /// # Arguments
    /// - `samples`: The iterator over time domain samples.
    /// - `subcarrier_functions`: The functions that take a complex value and returns the bits it corresponds to.
    /// - `cyclic_prefix_length`: The number of time samples to repeat from the end of the sample at the beginning as a cyclic prefix. Must be <= time domain symbol sample length.
    /// - `gain_factors`: The factor to scale each coefficient by.
    pub fn new(
        samples: I,
        subcarrier_decoders: [SubcarrierDecoder<'a, T>; CHANNELS_NUM],
        cyclic_prefix_length: usize,
        gain_factors: [Complex<T>; CHANNELS_NUM],
    ) -> Self {
        assert!(
            cyclic_prefix_length <= frequency_samples_to_time(subcarrier_decoders.len()),
            "Cyclic prefix can't be longer than the actual symbol."
        );
        OfdmDataDecoder {
            scratch_space: vec![T::zero(); frequency_samples_to_time(subcarrier_decoders.len())],
            subcarrier_decoders,
            samples,
            cyclic_prefix_length,
            gain_factors,
        }
    }

    /// Builds a new OFDM demodulation where each subcarrier's values are determined by the subcarrier's function.
    /// Uses the `samples` iterator to calculate the gain factors for the remainder of the frame.
    /// Assumes `samples` starts at the beginning of a frame's preamble.
    /// # Arguments
    /// - `samples`: The iterator over time domain samples.
    /// - `subcarrier_functions`: The functions that take a complex value and returns the bits it corresponds to.
    /// - `cyclic_prefix_length`: The number of time samples to repeat from the end of the symbol at the beginning as a cyclic prefix. Must be <= time domain symbol sample length.
    /// - `preamble_seed`: The seed used to generate a preamble.
    /// - `preamble_repeat_cnt`: The repeat count used to generate a preamble.
    pub fn build(
        mut samples: I,
        subcarrier_functions: [SubcarrierDecoder<'a, T>; CHANNELS_NUM],
        cyclic_prefix_length: usize,
        preamble_seed: u64,
        preamble_repeat_cnt: usize,
    ) -> Self {
        assert!(
            cyclic_prefix_length <= frequency_samples_to_time(subcarrier_functions.len()),
            "Cyclic prefix can't be longer than the actual symbol."
        );
        let symbol_time_len = frequency_samples_to_time(subcarrier_functions.len());

        // Get the gain factors of the nonzero pseudorandom noise symbol.
        let expected_preamble = ofdm_preamble_encode(
            preamble_seed,
            preamble_repeat_cnt,
            symbol_time_len,
            cyclic_prefix_length,
        )
        .collect();
        let gain_factors =
            gain_from_preamble(&mut samples, expected_preamble, cyclic_prefix_length);

        OfdmDataDecoder {
            scratch_space: vec![T::zero(); symbol_time_len],
            subcarrier_decoders: subcarrier_functions,
            samples,
            cyclic_prefix_length,
            gain_factors,
        }
    }

    /// Returns the number of positive frequency domain samples per symbol.
    pub fn freq_len(&self) -> usize {
        self.subcarrier_decoders.len()
    }

    /// Returns the time domain sample length per symbol without cyclic prefix.
    pub fn data_time_len(&self) -> usize {
        frequency_samples_to_time(self.subcarrier_decoders.len())
    }

    /// Returns the time domain sample length per symbol with cyclic prefix.
    pub fn prefixed_time_len(&self) -> usize {
        frequency_samples_to_time(self.subcarrier_decoders.len()) + self.cyclic_prefix_length
    }

    // Gain factor getter.
    pub fn gain_factors(&self) -> [Complex<T>; CHANNELS_NUM] {
        self.gain_factors
    }

    /// Returns the complex spectrum for the next symbol.
    pub fn next_complex(&mut self) -> Option<Vec<Complex<T>>> {
        // Remove cyclic prefix.
        for _ in 0..self.cyclic_prefix_length {
            self.samples.next()?;
        }

        // Load the next chunk of samples.
        for i in 0..self.data_time_len() {
            self.scratch_space[i] = self.samples.next()?;
        }

        // Use fft to convert back to complex values.
        Some(scaled_real_fft(&mut self.scratch_space))
    }
}

impl<'a, I: Iterator<Item = T>, T: FourierFloat, const CHANNELS_NUM: usize> Iterator
    for OfdmDataDecoder<'a, I, T, CHANNELS_NUM>
{
    type Item = Vec<bool>;

    /// Decodes the next symbol worth of samples into bits.
    fn next(&mut self) -> Option<Self::Item> {
        // Get the next complex values.
        let spectrum = self.next_complex()?;

        // Decode complex values to corresponding bit values.
        // Adjust by gain factor and apply channel's decoding function.
        let mut out = Vec::new();
        for (sample, channel_func) in spectrum
            .into_iter()
            .enumerate()
            .map(|(i, x)| x * self.gain_factors[i])
            .zip(self.subcarrier_decoders)
        {
            match channel_func {
                SubcarrierDecoder::Data(f) => out.extend_from_slice(&f(sample)),
                SubcarrierDecoder::Pilot(_) => todo!("Pilot decoding"),
            }
        }

        Some(out)
    }
}

impl<'a, I: Iterator<Item = D>, D: FourierFloat, const CHANNELS_NUM: usize>
    OfdmDataDecoder<'a, I, D, CHANNELS_NUM>
{
    /// Runs the iterator until exhausted and removes the guard bits.
    /// Function takes self by move because the decoder won't work again because the internal iterator will be exhausted.
    pub fn decode<T: bitvec::store::BitStore, O: bitvec::order::BitOrder>(
        decoder: Self,
    ) -> BitVec<T, O> {
        let mut decoded_bits = BitVec::<T, O>::from_iter(decoder.flatten())
            .into_iter()
            .rev()
            .skip_while(|x| !x) // skip 0's
            .skip(1) // skip guard
            .collect::<BitVec<T, O>>();
        decoded_bits.reverse();
        decoded_bits
    }
}

/// Generates an ofdm short training symbol consisting of repeated pseudorandom noise.
/// No cyclic prefix is added beyond the specified repeat cnt.
/// # Arguments
/// - `seed`: Seed for pseudorandom noise.
/// - `repeat_cnt`: Number of repetitions of the small symbol in the overall training symbol.
/// - `training_symbol_len`: The length in frequency domain of the training symbol.
fn ofdm_short_training_symbol<T: FourierFloat>(
    seed: u64,
    repeat_cnt: usize,
    training_symbol_len: usize,
) -> Vec<T> {
    scaled_real_ifft(
        &mut gen_pseudonoise_sequence(seed, training_symbol_len, (-1..=1).filter(|&x| x != 0))
            .enumerate()
            .map(|(i, x)| {
                if i == 0 || i == training_symbol_len - 1 {
                    (i, Complex::zero())
                } else {
                    let val = T::from(x).expect("Failed to convert.")
                        * T::from(2f64.sqrt()).expect("Failed to convert `2f64.sqrt()`.");
                    (i, Complex { re: val, im: val })
                }
            })
            .map(|(i, x)| {
                if i % repeat_cnt != 0 {
                    Complex::zero()
                } else {
                    x
                }
            })
            .collect::<Vec<_>>(),
    )
}

/// Generates nonzero pseudorandom noise of specified length.
/// Cyclic prefix is added before the training symbol length.
/// # Arguments
/// - `seed`: Seed for pseudorandom noise.
/// - `training_symbol_len`: The length in frequency domain of the training symbol.
/// - `cyclic_prefix_len`: The length in time domain of the cyclic prefix.
fn ofdm_long_training_symbol<T: FourierFloat>(
    seed: u64,
    training_symbol_len: usize,
    cyclic_prefix_len: usize,
) -> Vec<T> {
    let raw_time_symbol = scaled_real_ifft(
        &mut gen_pseudonoise_sequence(
            seed.wrapping_add(1),
            training_symbol_len,
            (-1..=1).filter(|&x| x != 0),
        )
        .enumerate()
        .map(|(i, x)| {
            if i == 0 || i == training_symbol_len - 1 {
                Complex::zero()
            } else {
                let val = T::from(x).expect("Failed to convert.")
                    * T::from(2f64.sqrt()).expect("Failed to convert `2f64.sqrt()`");
                Complex { re: val, im: val }
            }
        })
        .collect::<Vec<_>>(),
    );

    // Prepend cyclic prefix.
    let cyclic_prefix = &raw_time_symbol[raw_time_symbol.len() - cyclic_prefix_len..];
    let mut symbol_with_cyclic_prefix =
        Vec::with_capacity(raw_time_symbol.len() + cyclic_prefix_len);
    symbol_with_cyclic_prefix.extend_from_slice(cyclic_prefix);
    symbol_with_cyclic_prefix.extend_from_slice(&raw_time_symbol);
    symbol_with_cyclic_prefix
}

/// Generates an ofdm training preamble.
///
/// The preamble consists of two symbols:
/// 1. The first symbol is some pseudo random noise that repeats itself.
/// Repetition necessarily requires 0 values on frequencies that are not a multiple of the repetition count.
/// Autocorrelation can be used on this symbol because of the repetition.
/// 2. The second symbol is nonzero pseudo random noise on all frequencies.
/// This allows phase and amplitude corrections at the receiver for all subcarrier frequencies.
/// # Arguments
/// - `seed`: Seed for pseudo random noise.
/// - `repeat_cnt`: Number of repetitions of the small symbol in the overall training symbol.
/// - `training_symbol_len`: Length of each training sequence in time.
/// - `cyclic_prefix_len`: The length in time domain of the cyclic prefix.
/// # Reference
/// <https://ieeexplore.ieee.org/document/650240>
pub fn ofdm_preamble_encode<T: FourierFloat>(
    seed: u64,
    repeat_cnt: usize,
    training_symbol_len: usize,
    cyclic_prefix_len: usize,
) -> std::iter::Chain<std::vec::IntoIter<T>, std::vec::IntoIter<T>> {
    let training_symbol_len = time_samples_to_frequency(training_symbol_len);
    let short = ofdm_short_training_symbol(seed, repeat_cnt, training_symbol_len);
    // Change seed so two symbols don't autocorrelate. Probably unnecessary.
    let long =
        ofdm_long_training_symbol(seed.wrapping_add(1), training_symbol_len, cyclic_prefix_len);
    short.into_iter().chain(long)
}

/// Finds the starting position and timing metric of the first premable in the given slice if any exist.
/// This method uses auto-correlation.
/// # Arguments
/// - `samples`: The samples to find the preamble in.
/// - `repeating_premable_len`: The length of the expected repeated section in the preamble.
/// - `threshold`: The correlation threshold to determine the existence of a preamble.
/// - `correlation_window_len`: The length of the window used for correlation calculation.
pub fn ofdm_premable_auto_correlation_detector<T: FourierFloat + std::iter::Sum>(
    samples: &[Complex<T>],
    repeating_premable_len: usize,
    threshold: T,
    correlation_window_len: usize,
) -> Option<(usize, T)> {
    let mut highest_timing_metric = None;

    // Check each window to see if it is on average above the threshold.
    for (idx, timing_metric) in cross_correlation_timing_metric(
        samples,
        &samples[repeating_premable_len..],
        correlation_window_len,
    )
    .rolling_average(repeating_premable_len) // Should max at the exact sample start by averaging over the expected repeating length.
    .enumerate()
    {
        // If a preamble is detected
        if timing_metric > threshold {
            // Keep looking for the peak (higher values) until the peak is found (next value lower).
            // TODO smoothed z-score <https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362>
            // or other envelope detection.
            if timing_metric
                < highest_timing_metric
                    .unwrap_or((Default::default(), T::zero()))
                    .1
            {
                break;
            } else {
                highest_timing_metric = Some((idx, timing_metric))
            }
        }
    }

    highest_timing_metric
}

/// Computes the cross correlation at each offset in time of an unknown signal to a known reference signal.
pub fn cross_correlation_to_known_signal<'a, T: FourierFloat + std::iter::Sum>(
    unknown_signal: &'a [Complex<T>],
    known_signal: &'a [Complex<T>],
) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> T + Clone + 'a> {
    assert!(
        known_signal.len() < unknown_signal.len(),
        "Known signal should not be larger than unknown signal"
    );

    (0..unknown_signal.len() - known_signal.len()).map(move |i| {
        cross_correlation_timing_metric_single_value(
            &unknown_signal[i..=known_signal.len() + i],
            known_signal,
            known_signal.len(),
        )
    })
}

/// Finds the starting position and timing metric of the first premable in the given slice if any exist.
/// This method uses cross-correlation to a known signal.
/// As such it will correctly find the preamble start if the known signal occurs at the beginning of the preamble.
/// # Arguments
/// - `samples`: The samples to find the preamble in.
/// - `known_signal`: The known signal cross-correlation is attempting to find.
/// - `threshold`: The correlation threshold to determine the existence of a preamble.
pub fn ofdm_premable_cross_correlation_detector<T: FourierFloat + std::iter::Sum>(
    samples: &[Complex<T>],
    known_signal: &[Complex<T>],
    threshold: T,
) -> Option<(usize, T)> {
    let cross_correlation = cross_correlation_to_known_signal(samples, known_signal);
    for (i, correlation) in cross_correlation.enumerate() {
        if correlation > threshold {
            return Some((i, correlation));
        }
    }
    None
}

/// Iterator adapter that skips up to (but does not skip) the first detected value over the similarity threshold.
/// Consumes the iterator until the known signal has been consumed. Then reproduces the known signal and the remainder.
/// ```
/// use dsp::ofdm::SkipToKnownSignal;
/// let iter = [1.0, 2.0, 3.0, 4.0, 5.0].into_iter();
/// let known_signal = &[2.0, 3.0];
/// let skipped_iter = SkipToKnownSignal::new(iter, Box::new(*known_signal), 0.9);
/// assert_eq!(vec![2.0, 3.0, 4.0, 5.0], skipped_iter.collect::<Vec<_>>());
/// ```
#[derive(Debug, Clone)]
pub struct SkipToKnownSignal<I>
where
    I: Iterator,
{
    // Iterator being adapted.
    iter: I,
    /// The signal that is known. Complex for cross_correlation_calculation.
    known_signal: Box<[Complex<I::Item>]>,
    /// Threshold to determine detection.
    threshold: I::Item,
    /// If the iterator has skipped yet.
    skipped: bool,
    /// Values used to determine if there is a match.
    /// Also used to reproduce the part of the signal that is consumed for detection.
    buffered_values: Vec<Complex<I::Item>>,
}

impl<I: Iterator> SkipToKnownSignal<I>
where
    I::Item: Clone + Zero,
{
    pub fn new(iter: I, known_signal: Box<[I::Item]>, threshold: I::Item) -> Self {
        SkipToKnownSignal {
            iter,
            known_signal: known_signal
                .into_iter()
                .map(|x| Complex {
                    re: x.clone(),
                    im: I::Item::zero(),
                })
                .collect(),
            threshold,
            skipped: false,
            buffered_values: Vec::with_capacity(known_signal.len()),
        }
    }
}

impl<I: Iterator> Iterator for SkipToKnownSignal<I>
where
    I::Item: Sum + FourierFloat,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // If not already skipped to frame start then skip to frame start.
        if !self.skipped {
            // Fill the buffer with a number of values equal to the length of the known_signal.
            self.buffered_values.extend(
                self.iter
                    .by_ref()
                    .map(|x| Complex {
                        re: x,
                        im: I::Item::zero(),
                    })
                    .take(self.known_signal.len()),
            );

            // Repeatedly take a value and check if there is a sufficient correlation in the buffer to the known signal.
            // If there is then the frame has started.
            self.iter
                .by_ref()
                .take_while(|&x| {
                    let x = Complex {
                        re: x,
                        im: I::Item::zero(),
                    };
                    // If detected stop skipping.
                    let over_threshold = cross_correlation_timing_metric_single_value(
                        &self.buffered_values,
                        &self.known_signal,
                        self.known_signal.len(),
                    ) > self.threshold;
                    {
                        // Add the new value
                        self.buffered_values.remove(0);
                        self.buffered_values.push(x);
                        !over_threshold
                    }
                })
                .for_each(drop);
            self.skipped = true;
        }

        // Clear the buffer before consuming new iterator elements.
        if !self.buffered_values.is_empty() {
            Some(self.buffered_values.remove(0).re)
        } else {
            self.iter.next()
        }
    }
}

type OfdmFrame<'a, I, T> = std::iter::Chain<
    std::iter::Chain<vec::IntoIter<T>, vec::IntoIter<T>>,
    std::iter::Flatten<OfdmDataEncoder<'a, I, T>>,
>;

/// Generates an iterator of the ofdm preamble and data.
pub fn ofdm_frame_encoder<I: Iterator<Item = bool>, T: FourierFloat>(
    seed: u64,
    repeat_cnt: usize,
    ofdm_data_encoder: OfdmDataEncoder<'_, I, T>,
) -> OfdmFrame<'_, I, T> {
    ofdm_preamble_encode(
        seed,
        repeat_cnt,
        ofdm_data_encoder.time_len(),
        ofdm_data_encoder.cyclic_prefix_length,
    )
    .chain(ofdm_data_encoder.flatten())
}

/// Takes the expected preamble and compares it to the received preamble. Generates a scale factor for each subchannel to fix the gain.
fn gain_from_preamble<I, T, const CHANNELS_NUM: usize>(
    samples: &mut I,
    mut expected_preamble: Vec<T>,
    cyclic_prefix_len: usize,
) -> [Complex<T>; CHANNELS_NUM]
where
    I: Iterator<Item = T>,
    T: FourierFloat,
{
    let symbol_data_time_len: usize = frequency_samples_to_time(CHANNELS_NUM);

    // Skip the synchronization symbol and cyclic prefix.
    let expected_preamble = &mut expected_preamble[symbol_data_time_len + cyclic_prefix_len
        ..symbol_data_time_len + cyclic_prefix_len + symbol_data_time_len];
    samples
        .by_ref()
        .take(symbol_data_time_len + cyclic_prefix_len)
        .for_each(drop);

    // Determine the fourier coefficients of the expected preamble and the actual preamble coefficients.
    let mut actual_preamble = samples
        .by_ref()
        .take(symbol_data_time_len)
        .collect::<Vec<_>>();
    let actual_coefficients = scaled_real_fft(&mut actual_preamble);
    let expected_coefficients = scaled_real_fft(expected_preamble);
    let mut scale_factors = [Complex::one(); CHANNELS_NUM];

    assert_eq!(
        actual_coefficients.len(),
        expected_coefficients.len(),
        "Actual preamble coefficients length don't match expected length"
    );
    assert_eq!(
        CHANNELS_NUM,
        expected_coefficients.len(),
        "The number of channels doesn't match the length of the preamble."
    );

    // Compare the expected to actual to get the scale factors.
    for (i, (expected, actual)) in expected_coefficients
        .iter()
        .zip(actual_coefficients)
        .enumerate()
    {
        scale_factors[i] = expected / actual;
    }
    scale_factors
}

#[derive(Clone, Debug)]
pub struct OfdmFramesEncoder<'a, I, T>
where
    I: Iterator<Item = bool>,
    T: FourierFloat,
{
    bits: I,
    subcarrier_encoders: &'a [SubcarrierEncoder<bool, Complex<T>>],
    ofdm_spec: OfdmSpec,
}

impl<'a, I, T> OfdmFramesEncoder<'a, I, T>
where
    I: Iterator<Item = bool> + 'a,
    T: FourierFloat,
{
    pub fn new(
        bits: I,
        subcarrier_encoders: &'a [SubcarrierEncoder<bool, Complex<T>>],
        ofdm_spec: OfdmSpec,
    ) -> Self {
        OfdmFramesEncoder {
            bits,
            subcarrier_encoders,
            ofdm_spec,
        }
    }
}

impl<'a, I, T> Iterator for OfdmFramesEncoder<'a, I, T>
where
    T: FourierFloat,
    I: Iterator<Item = bool>,
{
    type Item = iter::Chain<
        iter::Chain<std::vec::IntoIter<T>, std::vec::IntoIter<T>>,
        iter::Flatten<OfdmDataEncoder<'a, std::vec::IntoIter<bool>, T>>,
    >;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO once cell here or initialize in constructor
        let bits_per_data_symbol = self
            .subcarrier_encoders
            .iter()
            .map(|&x| usize::from(x))
            .sum::<usize>();
        // TODO this allocation to a vector shouldn't be necessary. Fix by making it a lending iterator.
        let temp = self
            .bits
            .by_ref()
            .take(bits_per_data_symbol * self.ofdm_spec.data_symbols)
            .collect::<Vec<_>>();
        if temp.is_empty() {
            return None;
        }
        let encoder = OfdmDataEncoder::new(
            temp.into_iter(),
            self.subcarrier_encoders,
            self.ofdm_spec.cyclic_prefix_len,
        );
        Some(ofdm_frame_encoder(
            self.ofdm_spec.seed,
            self.ofdm_spec.short_training_repetitions,
            encoder,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct OfdmFramesDecoder<'a, I, T, const CHANNELS_NUM: usize>
where
    I: Iterator<Item = T>,
    T: FourierFloat,
{
    samples: I,
    subcarrier_decoders: [SubcarrierDecoder<'a, T>; CHANNELS_NUM],
    ofdm_spec: OfdmSpec,
}

impl<'a, I, T, const CHANNELS_NUM: usize> OfdmFramesDecoder<'a, I, T, CHANNELS_NUM>
where
    I: Iterator<Item = T> + 'a,
    T: FourierFloat,
{
    pub fn new(
        samples: I,
        subcarrier_decoders: [SubcarrierDecoder<'a, T>; CHANNELS_NUM],
        ofdm_spec: OfdmSpec,
    ) -> Self {
        Self {
            samples,
            subcarrier_decoders,
            ofdm_spec,
        }
    }
}

impl<'a, I, T, const CHANNELS_NUM: usize> Iterator for OfdmFramesDecoder<'a, I, T, CHANNELS_NUM>
where
    T: FourierFloat,
    I: Iterator<Item = T>,
{
    type Item = OfdmDataDecoder<'a, std::vec::IntoIter<T>, T, CHANNELS_NUM>;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO once cell here
        let preamble_sample_len = ofdm_preamble_encode::<T>(
            self.ofdm_spec.seed,
            self.ofdm_spec.short_training_repetitions,
            self.ofdm_spec.time_symbol_len,
            self.ofdm_spec.cyclic_prefix_len,
        )
        .count();
        // The extra partial symbol is compinsated with a 1 as below.
        // TODO fix this incorrect behavior. The partial symbol shouldn't be an extra symbol it should be the last in to bring the number of symbols to the correct amount.
        // This should be fixed the the OfdmDataEncoder
        let data_sample_len = (self.ofdm_spec.data_symbols + 1)
            * (self.ofdm_spec.time_symbol_len + self.ofdm_spec.cyclic_prefix_len);

        // TODO vec allocation might be unnecessary.
        let temp = self
            .samples
            .by_ref()
            .take(preamble_sample_len + data_sample_len)
            .collect::<Vec<_>>();
        if temp.is_empty() {
            return None;
        };

        Some(OfdmDataDecoder::build(
            temp.into_iter(),
            self.subcarrier_decoders,
            self.ofdm_spec.cyclic_prefix_len,
            self.ofdm_spec.seed,
            self.ofdm_spec.short_training_repetitions,
        ))
    }
}
