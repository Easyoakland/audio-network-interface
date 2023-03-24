use realfft::RealFftPlanner;

/// Take normalized fft and return magnitudes.
pub fn scaled_fft(data: Vec<f64>) -> Vec<f64> {
    // To get normalized results, each element must be scaled by 1/sqrt(length).
    // If the processing involves both an FFT and an iFFT step, it is advisable to merge the two normalization steps to a single, by scaling by 1/length.
    // So overall formula for an element is `|x/sqrt(length)|`
    let scale_factor = 1.0 / (data.len() as f64).sqrt();
    fft(data).into_iter().map(|v| v * scale_factor).collect()
}

/// Take unnormalized fft and return magnitudes.
pub fn fft(mut data: Vec<f64>) -> Vec<f64> {
    let mut real_planner = RealFftPlanner::<f64>::new();

    // Create a FFT.
    let r2c = real_planner.plan_fft_forward(data.len());
    // Make output vector. `spectrum.len() == length /2 + 1`
    let mut spectrum = r2c.make_output_vec();

    // Forward transform the input data
    r2c.process(&mut data, &mut spectrum).unwrap();

    let mut out = Vec::with_capacity(spectrum.len());
    out.extend(spectrum.iter().map(|v| v.norm()));
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

    pub const fn rectangular(_n: usize, _samples: usize) -> WindowFloat {
        1.0
    }

    pub fn hamming(n: usize, samples: usize) -> WindowFloat {
        const A0: WindowFloat = 0.53836;
        A0 - (1.0 - A0)
            * WindowFloat::cos((2.0 * PI * n as WindowFloat) / samples as WindowFloat - 1.0)
    }

    pub fn hann(n: usize, samples: usize) -> WindowFloat {
        const A0: WindowFloat = 0.5;
        A0 * (1.0
            - WindowFloat::cos((2.0 * PI * n as WindowFloat) / (samples as WindowFloat - 1.0)))
    }

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
