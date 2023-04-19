//! Plots the frequencies corresponding the data channels over time.
//! Useful for analyzing why a bit is misinterpreted on the receiver.

use audio_network_interface::{
    args::{TransmissionCli, TransmissionSpec},
    file_io::read_wav,
    plotting::plot_series,
};
use clap::Parser as _;
use std::{ffi::OsString, path::Path};
use stft::{fft::window_fn, SpecCompute, WindowLength};

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = TransmissionCli::parse();
    simple_logger::init_with_level(opt.base.log_opt.log_level).unwrap();

    match opt.transmission_spec {
        TransmissionSpec::Fdm(fdm_spec) => {
            // Read in wav file.
            let (spec, data) = read_wav(&Path::new(&opt.base.file_opt.in_file));

            // Set window parameters so each symbol is one sample.
            let window_len = WindowLength::from_duration(
                std::time::Duration::from_millis(fdm_spec.symbol_time),
                spec.sample_rate as f32,
            );
            let window_step = window_len;

            // Compute stft of data.
            let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn::hann);
            let frequency_analysis = spec_compute.stft();

            for f in (1..=fdm_spec.parallel_channels)
                .map(|i| (fdm_spec.start_freq + i as f32 * fdm_spec.bit_width))
            {
                let transient = frequency_analysis
                    .get_bin(f, spec.sample_rate as f32)
                    .unwrap();

                // Rename input file, change to .png, and append `_out` to basename.
                let file_out = {
                    let parent = Path::new(&opt.base.file_opt.in_file).parent().unwrap();
                    let mut temp = Path::new(&opt.base.file_opt.in_file)
                        .file_stem()
                        .unwrap()
                        .to_owned();
                    temp.push(OsString::from(format!("_time_vs_channel_{f}.png")));
                    parent.join(temp)
                };

                plot_series(
                    file_out.to_str().unwrap(),
                    vec![transient.clone()],
                    "transmission hz vs sliding window",
                    false,
                    -5.0,
                )?;
            }
        }
        _ => panic!("Incorrect transmission spec."),
    }

    Ok(())
}
