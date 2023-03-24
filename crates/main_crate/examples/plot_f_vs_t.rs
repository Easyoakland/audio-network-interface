//! Plots the frequencies corresponding the data channels over time.
//! Useful for analyzing why a bit is misinterpreted on the receiver.

use audio_network_interface::{
    args::transmit_receive_args::Args,
    file_io::read_wav,
    plotting::{generate_constant_series, plot_series},
};
use clap::Parser as _;
use iterator_adapters::IteratorAdapter;
use ordered_float::OrderedFloat;
use std::{ffi::OsString, path::Path};
use stft::{fft::window_fn, SpecCompute, WindowLength};

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = Args::parse();
    simple_logger::init_with_level(opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = read_wav(&Path::new(&opt.in_file));

    // Set window parameters for high resolution.
    // Bin width is `Fs/N` where `Fs` is sampling frequency and `N` is samples.
    // let window_len = WindowLength::from_samples(2usize.pow(11));
    // let window_step = window_len / 4;

    // Set window parameters so each symbol is one sample.
    let window_len = WindowLength::from_duration(
        std::time::Duration::from_millis(opt.symbol_time),
        spec.sample_rate as f32,
    );
    let window_step = window_len;

    // Compute stft of data.
    let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn::hann);
    let frequency_analysis = spec_compute.stft();

    for f in (1..=opt.parallel_channels).map(|i| (opt.start_freq + i as f32 * opt.bit_width)) {
        let transient = frequency_analysis
            .get_bin(f, spec.sample_rate as f32)
            .unwrap();

        // Rename input file, change to .png, and append `_out` to basename.
        let file_out = {
            let parent = Path::new(&opt.in_file).parent().unwrap();
            let mut temp = Path::new(&opt.in_file).file_stem().unwrap().to_owned();
            temp.push(OsString::from(format!("_time_vs_channel_{f}.png")));
            parent.join(temp)
        };

        let derivative = transient
            .windows(2)
            .map(|s| (s[0], s[1]))
            .map(|(y1, y2)| (y2 - y1));
        let derivative_beginning_of_signal =
            derivative
                .clone()
                .enumerate()
                .fold(
                    (0, 0.0),
                    |acc, sample| {
                        if sample.1 > 10.0 * acc.1 {
                            sample
                        } else {
                            acc
                        }
                    },
                );
        let derivative_end_of_signal =
            derivative
                .clone()
                .enumerate()
                .rev()
                .fold((0, 0.0), |acc, sample| {
                    if -sample.1 > -10.0 * acc.1 {
                        sample
                    } else {
                        acc
                    }
                });
        let derivative_signal_extents = derivative
            .clone()
            .enumerate()
            .map(|(i, x)| {
                if i == derivative_beginning_of_signal.0 || i == derivative_end_of_signal.0 {
                    x.abs()
                } else {
                    0.0
                }
            })
            .collect();

        // let signal_high_1 =
        // generate_constant_series(transient[derivative_beginning_of_signal.0], transient.len())
        // .collect();
        // let signal_high_2 = generate_constant_series(
        // transient[derivative_beginning_of_signal.0 + 1],
        // transient.len(),
        // )
        // .collect();
        // let signal_high_av = generate_constant_series(
        // (transient[derivative_beginning_of_signal.0 + 1]
        // + transient[derivative_beginning_of_signal.0])
        // / 2.0,
        // transient.len(),
        // )
        // .collect();
        // let max = *transient
        // .iter()
        // .max_by(|x, y| x.partial_cmp(y).unwrap())
        // .unwrap();
        let max_90th = transient.iter().map(|x| OrderedFloat(*x)).kth_order(90).0;
        plot_series(
            &opt.in_file,
            file_out.to_str().unwrap(),
            vec![
                transient.clone(),
                generate_constant_series(
                    transient
                        .clone()
                        .into_iter()
                        .map(|x| ordered_float::OrderedFloat(x))
                        .median()
                        .0,
                    transient.len(),
                )
                .collect(),
                // frequency_analysis.power(),
                derivative_signal_extents,
                // signal_high_1,
                // signal_high_2,
                // signal_high_av,
                // generate_constant_series(transient.iter().copied().mean(), transient.len())
                // .collect(),
                // generate_constant_series(max / 2.0, transient.len()).collect(),
                generate_constant_series(max_90th / 2.0, transient.len()).collect(),
            ],
            "transmission hz vs sliding window",
            false,
            -5.0,
        )?;
    }
    Ok(())
}
