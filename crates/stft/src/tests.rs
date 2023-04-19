use crate::{bin_width_from_time, fft::window_fn, SpecCompute, WindowLength};
use proptest::{prelude::ProptestConfig, proptest};

fn test_signal(mut signal_frequency: f32, sample_rate: f32) {
    // Signal frequency must be less than Nyquist frequency
    while signal_frequency > sample_rate / 2.0 {
        signal_frequency /= 2.0
    }
    assert!(
        signal_frequency > 0.0,
        "If signal frequency is 0 test signal will have no power."
    );

    let signal_frequency = signal_frequency;
    // dbg!(signal_frequency, sample_rate);

    // Generate example signal window parameters.
    let window_len = WindowLength::from_samples(2usize.pow(3));
    let window_step = window_len / 4;
    let window_fn = window_fn::rectangular;
    // Generate samples.
    let mut data = Vec::new();
    for sample_num in 0..20000 {
        let t = sample_num as f32 / sample_rate;
        data.push((2.0 * std::f32::consts::PI * signal_frequency * t).sin() as f64);
    }
    let bin_width = bin_width_from_time(sample_rate, window_len.samples());

    // Compute the stft.
    let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn);
    let frequency_analysis = spec_compute.stft();

    // Check for zero lengths.
    assert!(
        !frequency_analysis.data.is_empty(),
        "No frequencies in frequency analysis."
    );
    for transient in frequency_analysis.data.iter() {
        assert!(!transient.is_empty(), "Transient length is 0.");
    }

    let power_transient = frequency_analysis.power();
    let total_energy: f64 = power_transient.into_iter().sum();
    // eprintln!("Total signal energy for all frequencies: {total_energy}");
    let detection_threshold = 0.15 * total_energy;

    // Make sure no frequency has more energy than all frequencies combined.
    for (bin_idx, transient) in frequency_analysis.data.iter().enumerate() {
        let freq_lower = bin_width * bin_idx as f32;
        let freq_higher = freq_lower + bin_width;
        let total_energy_of_freq = transient.iter().map(|x| x.powi(2)).sum::<f64>();
        // eprintln!("Bin index: {bin_idx}; Frequency: {freq_lower}-{freq_higher} has energy {total_energy_of_freq}. Total energy {total_energy_of_freq}");
        assert!(total_energy_of_freq <= total_energy, "Bin index: {bin_idx}; Frequency: {freq_lower}-{freq_higher} has energy {total_energy_of_freq} which is more than total signal's {total_energy}.");
    }

    // Make sure the correct frequencies are identified.
    for (bin_idx, transient) in frequency_analysis.data.iter().enumerate() {
        let freq_lower = bin_width * bin_idx as f32;
        let freq_higher = freq_lower + bin_width;
        let total_energy_of_freq = transient.iter().map(|x| x.powi(2)).sum::<f64>();
        // eprintln!("Bin width: {bin_width}; Bin index: {bin_idx}; Frequency: {freq_lower}-{freq_higher}; Total energy: {total_energy_of_freq}");

        // If the frequency is in the signal then it should have a large amplitude.
        if freq_lower <= signal_frequency && signal_frequency < freq_higher - f32::EPSILON {
            assert!(total_energy_of_freq > detection_threshold, "Frequency {freq_lower}-{freq_higher} with energy {total_energy_of_freq} <= than detection threshold {detection_threshold}.");
        }
        // Otherwise it should have a small amplitude.
        else {
            assert!(total_energy_of_freq < detection_threshold, "Frequency {freq_lower}-{freq_higher} with energy {total_energy_of_freq} >= than detection threshold {detection_threshold}.");
        }
    }
}

#[test]
fn hz_60_sin_wave() {
    test_signal(60.0, 20_000.0);
}

#[test]
fn test_min_signal() {
    test_signal(1.0, 1.0);
}

fn test_signal_for_panics(signal_frequency: f32, sample_rate: f32) {
    // Reassign invalid input.
    // Must have positive frequency.
    let mut signal_frequency = signal_frequency.abs();
    // Must have positive sample rate.
    let mut sample_rate = sample_rate.abs();
    // Must have nonzero sample rate.
    if sample_rate <= f32::EPSILON {
        sample_rate = 1.0
    }
    // If signal frequency is 0 the signal will not be detected because time data will be [0.0,0.0,...]
    if signal_frequency <= f32::EPSILON {
        signal_frequency = 1.0
    }
    // Signal frequency must be less than Nyquist frequency
    while signal_frequency > sample_rate / 2.0 {
        signal_frequency /= 2.0
    }

    let signal_frequency = signal_frequency;

    // Generate example signal window parameters.
    let window_len = WindowLength::from_samples(2usize.pow(3));
    let window_step = window_len / 4;
    let window_fn = window_fn::rectangular;
    // Generate samples.
    let mut data = Vec::new();
    for sample_num in 0..20000 {
        let t = sample_num as f32 / sample_rate;
        data.push((2.0 * std::f32::consts::PI * signal_frequency * t).sin() as f64);
    }
    let bin_width = bin_width_from_time(sample_rate, window_len.samples());

    // Compute the stft.
    let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn);
    let frequency_analysis = spec_compute.stft();

    // Check for zero lengths.
    assert!(
        !frequency_analysis.data.is_empty(),
        "No frequencies in frequency analysis."
    );
    for transient in frequency_analysis.data.iter() {
        assert!(!transient.is_empty(), "Transient length is 0.");
    }

    let power_transient = frequency_analysis.power();
    let total_energy: f64 = power_transient.into_iter().sum();

    // Make sure no frequency has more energy than all frequencies combined.
    for (bin_idx, transient) in frequency_analysis.data.iter().enumerate() {
        let freq_lower = bin_width * bin_idx as f32;
        let freq_higher = freq_lower + bin_width;
        let total_energy_of_freq = transient.iter().map(|x| x.powi(2)).sum::<f64>();
        assert!(total_energy_of_freq <= total_energy, "Bin index: {bin_idx}; Frequency: {freq_lower}-{freq_higher} has energy {total_energy_of_freq} which is more than total signal's {total_energy}.");
    }
}

fn float_eq(a: f64, b: f64, epsilon: f64) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();

    if a == b {
        // shortcut, handles infinities
        true
    } else if a == 0.0 || b == 0.0 || diff < epsilon {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        diff < epsilon * epsilon
    } else {
        // use relative error
        diff / (abs_a + abs_b) < dbg!(epsilon)
    }
}

fn power_freq_eq_time(mut signal_frequency: f32, sample_rate: f32) {
    // Signal frequency must be less than Nyquist frequency
    while signal_frequency > sample_rate / 2.0 {
        signal_frequency /= 2.0
    }

    // Generate signal window parameters.
    let window_len = WindowLength::from_samples(2usize.pow(3));
    let window_step = window_len / 1;
    let window_fn = window_fn::rectangular;
    // Generate samples.
    let mut data = Vec::new();
    for sample_num in 0..20000 {
        let t = sample_num as f32 / sample_rate;
        data.push((2.0 * std::f32::consts::PI * signal_frequency * t).sin() as f64);
    }

    // Compute the energy of the time series and the frequency series and compare.
    let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn);
    let time_energy: f64 = spec_compute.power().into_iter().sum();
    let freq_energy: f64 = spec_compute.stft().power().into_iter().sum();
    assert!(
        float_eq(freq_energy, time_energy, 10e-5),
        "Time energy {time_energy} != Frequency energy {freq_energy}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))] // Decrease case default from 256 to 10 because these test are slow.
    #[test]
    fn proptest_single_frequency(signal_frequency: f32, sample_rate: f32) {
        test_signal_for_panics(signal_frequency, sample_rate);
    }

    // TODO make below pass. Energy leaks to adjacent bins causing fail sometimes.
    // #[test]
    // fn proptest_input_space(signal_frequency in 2u32.., sample_rate in 2u32..) {
        // test_signal(signal_frequency as f32, sample_rate as f32);
    // }

    #[test]
    fn proptest_power_freq_eq_time(signal_frequency in 2u32.., sample_rate in 2u32..) {
        power_freq_eq_time(signal_frequency as f32, sample_rate as f32);
    }
}
