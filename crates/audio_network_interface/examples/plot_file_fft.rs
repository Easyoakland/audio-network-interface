//! Plots the dtft frequency spectrum of the intput file.

use audio_network_interface::{args::BaseCli, file_io::read_wav, plotting::plot_fft};
use clap::Parser as _;
use futures_lite::future::block_on;
use log::info;
use stft::fft;

fn main() -> Result<(), anyhow::Error> {
    block_on(async_main())
}

async fn async_main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = BaseCli::parse();
    simple_logger::init_with_level(opt.log_opt.log_level).unwrap();

    // Read in wav file.
    let (spec, mut data) = read_wav(&opt.file_in.in_file).await?;

    // Bin width is `Fs/N`, `MaxFreq = Fs/2`; where `Fs` is sampling frequency and `N` is samples.
    let bin_width = (spec.sample_rate as f32) / data.len() as f32;
    // Take fft of data.
    let data = fft::scaled_real_fft(&mut data)
        .into_iter()
        .map(|x| x.norm())
        .collect();

    info!("Plotting fft with bin width: {bin_width}.");
    plot_fft(
        &opt.file_out.out_file,
        data,
        spec.sample_rate as f32,
        bin_width,
    )?;

    Ok(())
}
