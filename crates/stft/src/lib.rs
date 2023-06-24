//! Crate for finding the Short Time Fourier Transform of a real signal.

use derive_more::{Add, Div, Mul, Sub};
use fft::window_fn::WindowFn;
use std::time::Duration;

/// Relating to frequency analysis ex. dtft and fft.
pub mod fft;

/// A window length. Wraps the number of samples with methods for converting to/from time.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Add, Div, Mul, Sub)]
pub struct WindowLength {
    sample_num: usize,
}

impl WindowLength {
    /// [`WindowLength`] constructor.
    #[must_use]
    pub fn from_samples(sample_num: usize) -> Self {
        WindowLength { sample_num }
    }

    /// [`WindowLength`] constructor.
    /// # Arguments
    /// - `duration`: Length of the window in time.
    /// - `sample_rate`: Samples per second.
    #[must_use]
    pub fn from_duration(duration: Duration, sample_rate: f32) -> Self {
        // secs * samples/sec = samples
        WindowLength {
            sample_num: (duration.as_secs_f32() * sample_rate) as usize,
        }
    }

    /// Getter for samples
    #[must_use]
    pub fn samples(&self) -> usize {
        self.sample_num
    }

    /// Converts samples to duration. Samples / (samples/sec)= secs
    /// # Arguments
    /// - `sample_rate`: the samples per second.
    #[must_use]
    pub fn duration(&self, sample_rate: f32) -> Duration {
        Duration::from_secs_f32(self.sample_num as f32 / sample_rate)
    }
}

/// `window*(time_samples/window)*(1/(time_sample/sec)) = sec`
/// TODO add test
#[must_use]
pub fn window_to_time(window_step: usize, window_idx: usize, sample_rate: f32) -> f32 {
    window_idx as f32 * window_step as f32 * (1.0 / sample_rate)
}

/// `sec*(time_sample/sec)*(1/(time_samples/window)) = window`
/// TODO add test
#[must_use]
pub fn time_to_window(sample_rate: f32, time: f32, window_step: usize) -> f32 {
    time * sample_rate * (1.0 / window_step as f32)
}

/// The information required to calculate a spectrograph.
#[derive(Debug)]
pub struct SpecCompute {
    data: Vec<f64>,            // Time domain data.
    window_len: WindowLength,  // The length of a window.
    window_step: WindowLength, // The step length used for each successive window. Should be at most the length of the window_step.
    window_fn: WindowFn,       // The window function to use.
}

/// The calculated spectrograph.
// TODO unify into one vector with accessor methods and stride.
#[derive(Debug)]
pub struct Stft {
    data: Vec<Vec<f64>>, // Vector of each frequency's individual time varying amplitude.
}

impl Stft {
    /// Get the transient analysis for the frequency bin that contains the indicated frequency if it exists.
    #[must_use]
    pub fn get_bin(&self, frequency: f32, sample_rate: f32) -> Option<&Vec<f64>> {
        self.data
            .get((frequency / bin_width_from_freq(sample_rate, self.data.len())).trunc() as usize)
    }

    /// Get the frequency bins for a time closest from below.
    /// TODO add test
    #[must_use]
    pub fn get_time(&self, sample_rate: f32, time: f32, window_step: usize) -> Option<Vec<f64>> {
        assert!(sample_rate > 0., "sample rate should be a positive value");
        assert!(time >= 0., "time should be a positive value");
        let window = time_to_window(sample_rate, time, window_step) as usize;
        self.data[0].get(window)?; // Check validity before iteration.
        Some(
            (0..self.data[0].len())
                .map(|f| self.data[f][window])
                .collect(),
        )
    }

    /// Number of frequency bins.
    #[must_use]
    pub fn bin_cnt(&self) -> usize {
        self.data.len()
    }

    /// Number of windows of time.
    #[must_use]
    pub fn window_cnt(&self) -> usize {
        self.data[0].len()
    }

    /// Calculates the power of the signal per time.
    #[must_use]
    pub fn power(&self) -> Vec<f64> {
        let mut power_transient = Vec::with_capacity(self.data[0].len());
        for time in 0..self.data[0].len() {
            let mut spectral_density = Vec::with_capacity(self.data.len());
            for f in 0..self.data.len() {
                // Spectral density is |X(w)|^2 by Parseval's theorem.
                let mut power = self.data[f][time] * self.data[f][time];

                // All powers of a real valued signal are split between the positive and negative except 0 and max frequency.
                // Double these values to preserve the power since negative frequencies are ignored.
                if f != 0 && f != self.data.len() - 1 {
                    power *= 2.0;
                }
                spectral_density.push(power);
            }
            // Sum spectral density over all frequencies to get total energy of the period, and power is energy/period.
            power_transient.push(spectral_density.into_iter().sum());
        }
        power_transient
    }

    /// Each bin's transient value iterating from 0 hz to Nyquist frequency.
    pub fn bins(&self) -> impl Iterator<Item = Vec<f64>> + '_ {
        (0..self.data.len()).map(|bin| self.data[bin].clone())
    }

    /// Each `window_length`'s bins from the first window at t=0 to the last window.
    pub fn times(&self) -> impl Iterator<Item = Vec<f64>> + '_ {
        (0..self.data[0].len()).map(|time| // For each window
                (0..self.data.len()).map(|f| // Collect all frequency bins
                     self.data[f][time]).collect())
    }
}

/// Calculates bin width of a dtft from the number of time samples.
/// Bin width is `Fs/N` where `Fs` is sampling frequency and `N` is samples.
#[must_use]
pub fn bin_width_from_time(sample_rate: f32, sample_cnt: usize) -> f32 {
    sample_rate / sample_cnt as f32
}

/// Calculates bin width of a dtft from the number of frequencies.
/// Bin width is `Max_Frequency / number of frequencies`
#[must_use]
pub fn bin_width_from_freq(sample_rate: f32, sample_num: usize) -> f32 {
    let max_freq = sample_rate / 2.0;
    max_freq / sample_num as f32
}

/// Converts a number of frequency samples to time samples.
/// N samples to 2(N-1) samples.
#[must_use]
pub const fn frequency_samples_to_time(freq_samples: usize) -> usize {
    2 * (freq_samples - 1)
}

/// Converts a number of time samples to frequency samples.
/// N samples to (N/2)+1 samples.
#[must_use]
pub const fn time_samples_to_frequency(time_samples: usize) -> usize {
    (time_samples / 2) + 1
}

impl SpecCompute {
    /// Basic constructor.
    pub fn new(
        data: Vec<f64>,
        window_len: WindowLength,
        window_step: WindowLength,
        window_fn: WindowFn,
    ) -> Self {
        assert!(
            window_step <= window_len,
            "Step length should not be larger than the window itself."
        );
        SpecCompute {
            data,
            window_len,
            window_step,
            window_fn,
        }
    }

    /// Calculates the power of the signal per time.
    #[must_use]
    pub fn power(&self) -> Vec<f64> {
        self.data.iter().map(|x| x * x).collect()
    }

    /// Returns the time varying frequency analysis of frequencies.
    /// Outer vec is each frequency. Inner vec is each value per window step.
    #[must_use]
    pub fn stft(&self) -> Stft {
        // Given 2n time samples get n+1 frequency samples.
        let mut result = vec![vec![]; self.window_len.samples() / 2 + 1];

        // For the shifting window.
        for window_of_data in self
            .data
            // Take a window of samples at a time.
            .windows(self.window_len.samples())
            // Step by the window step.
            .step_by(self.window_step.samples())
        {
            // Apply windowing function.
            let mut window_of_data = window_of_data.to_owned();
            fft::window_fn::apply_window(&mut window_of_data, self.window_fn);

            // Take the fourier transform of the window.
            let freq_data = fft::scaled_real_fft(&mut window_of_data);

            // Add the analysis to the results.
            for (bin_idx, bin_value) in freq_data.into_iter().enumerate() {
                result[bin_idx].push(bin_value.norm());
            }
        }

        Stft { data: result }
    }

    /// Getter for time data
    #[must_use]
    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }
}

impl Stft {
    /// Getter for frequency analysis data.
    #[must_use]
    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }
}

#[cfg(test)]
mod tests;
