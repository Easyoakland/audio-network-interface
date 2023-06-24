use realfft::{
    num_complex::Complex,
    num_traits::{Float, FromPrimitive, Signed},
    RealFftPlanner,
};
use std::{
    fmt::Debug,
    marker::{Copy, Send, Sync},
};

// Trait acts as generic bounds alias.
#[rustfmt::skip]
pub trait FourierFloat: Copy + Float  + FromPrimitive + Signed + Sync + Send + Debug + 'static {}
#[rustfmt::skip]
impl<T: Copy + Float + FromPrimitive + Signed + Sync + Send + Debug + 'static> FourierFloat for T {}

/// Take normalized fft and return complex values.
/// Input becomes garbage after this.
pub fn scaled_real_fft<T: FourierFloat>(time: &mut [T]) -> Vec<Complex<T>> {
    // To get normalized results, each element must be scaled by 1/sqrt(length).
    // If the processing involves both an FFT and an iFFT step, it is advisable to merge the two normalization steps to a single, by scaling by 1/length.
    // So overall formula for an element is `|x/sqrt(length)|`
    let scale_factor =
        T::from_f64(1.0 / (time.len() as f64).sqrt()).expect("Can't convert to f64.");
    real_fft(time)
        .into_iter()
        .map(|v| v * scale_factor)
        .collect()
}

/// Take unnormalized fft and return complex values.
/// Input becomes garbage after this.
pub fn real_fft<T: FourierFloat>(time: &mut [T]) -> Vec<Complex<T>> {
    let mut real_planner = RealFftPlanner::<T>::new();

    // Create a FFT.
    let r2c = real_planner.plan_fft_forward(time.len());
    // Make output vector. `spectrum.len() == length / 2 + 1`
    let mut spectrum = r2c.make_output_vec();

    // Forward transform the input data
    r2c.process(time, &mut spectrum).unwrap();

    let mut out = Vec::with_capacity(spectrum.len());
    out.extend(spectrum);
    out
}

/// Take normalized ifft and return magnitudes.
/// Input vector becomes garbage after this.
/// Keep in mind that this is the dc through positive frequencies up to the Nyquist frequency.
/// Goes from M frequency values to 2(M-1) time values.
pub fn scaled_real_ifft<T: FourierFloat>(spectrum: &mut [Complex<T>]) -> Vec<T> {
    // To get normalized results, each element must be scaled by 1/sqrt(length).
    // This length refers to the time domain length N and not the frequency domain length N/2+1.
    // If the processing involves both an FFT and an iFFT step, it is advisable to merge the two normalization steps to a single, by scaling by 1/length.
    // So overall formula for an element is `|x/sqrt(2*(length-1))|`
    let scale_factor = 1.0 / (2.0 * (spectrum.len() - 1) as f64).sqrt();
    let scale_factor = T::from_f64(scale_factor).expect("Can't convert to float type.");
    real_ifft(spectrum)
        .into_iter()
        .map(|v| v * scale_factor)
        .collect()
}

/// Take unnormalized ifft and return magnitudes.
/// Input vector becomes garbage after this.
/// Keep in mind that this is the dc through positive frequencies up to the Nyquist frequency. Except for the 0th and Nyquist their power is doubled because they represent power of positive and negative frequencies.
pub fn real_ifft<T: FourierFloat>(spectrum: &mut [Complex<T>]) -> Vec<T> {
    let mut real_planner = RealFftPlanner::<T>::new();

    // Create a FFT.
    let c2r = real_planner.plan_fft_inverse(2 * (spectrum.len() - 1));
    // Make output vector. `2*(spectrum.len() - 1) == length`
    let mut time = c2r.make_output_vec();

    // Inverse transform the input data
    c2r.process(spectrum, &mut time).unwrap();

    let mut out = Vec::with_capacity(time.len());
    out.extend(time.iter());
    out
}

/// Windowing functions useful for dtft analysis. See <https://en.wikipedia.org/wiki/Window_function> for details.
pub mod window_fn {
    pub type WindowFloat = f64;
    pub type WindowFn = fn(usize, usize) -> WindowFloat;
    use std::f64::consts::PI;

    /// Applies the given window function to the input data.
    pub fn apply_window(data: &mut [WindowFloat], window: WindowFn) {
        let data_len = data.len();
        for (i, elem) in data.iter_mut().enumerate() {
            *elem *= window(i, data_len);
        }
    }

    #[must_use]
    pub const fn rectangular(_n: usize, _samples: usize) -> WindowFloat {
        1.0
    }

    #[must_use]
    pub fn hamming(n: usize, samples: usize) -> WindowFloat {
        const A0: WindowFloat = 0.53836;
        A0 - (1.0 - A0)
            * WindowFloat::cos((2.0 * PI * n as WindowFloat) / samples as WindowFloat - 1.0)
    }

    #[must_use]
    pub fn hann(n: usize, samples: usize) -> WindowFloat {
        const A0: WindowFloat = 0.5;
        A0 * (1.0
            - WindowFloat::cos((2.0 * PI * n as WindowFloat) / (samples as WindowFloat - 1.0)))
    }

    #[must_use]
    pub fn blackman_harris(n: usize, samples: usize) -> WindowFloat {
        const A0: WindowFloat = 0.35875;
        const A1: WindowFloat = 0.48829;
        const A2: WindowFloat = 0.14128;
        const A3: WindowFloat = 0.01168;

        let arg = 2.0 * PI * n as WindowFloat / (samples as WindowFloat - 1.0);

        A0 - A1 * WindowFloat::cos(arg) + A2 * WindowFloat::cos(2.0 * arg)
            - A3 * WindowFloat::cos(3.0 * arg)
    }
}
