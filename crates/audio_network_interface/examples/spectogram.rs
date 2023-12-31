//! Plots a spectogram of the input data using the same methods used for actual analysis.
//! Good check that analysis is making sense.

use audio_network_interface::{args::BaseCli, file_io::read_wav};
use clap::Parser as _;
use futures_lite::future::block_on;
use stft::{fft::window_fn, SpecCompute, WindowLength};

fn main() -> Result<(), anyhow::Error> {
    block_on(async_main())
}

async fn async_main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = BaseCli::parse();
    simple_logger::init_with_level(opt.log_opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = read_wav(&opt.file_in.in_file).await?;

    // Bin width is `Fs/N` where `Fs` is sampling frequency and `N` is samples.
    let window_len = WindowLength::from_samples(2usize.pow(11));
    let bin_width = stft::bin_width_from_time(spec.sample_rate as f32, window_len.samples());

    let spec_compute = SpecCompute::new(data, window_len, window_len / 4, window_fn::hann);
    let frequency_analysis = spec_compute.stft();

    audio_network_interface::plotting::plot_spectogram(
        &frequency_analysis,
        bin_width,
        &opt.file_out.out_file,
        "Frequency vs Time Spectogram",
        false,
    )?;

    Ok(())
}
