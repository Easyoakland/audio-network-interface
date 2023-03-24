//! Plots the dtft frequency spectrum of the intput file.

use audio_network_interface::{
    args::transmit_receive_args::Args, file_io::read_wav, plotting::plot_fft,
};
use clap::Parser as _;
use log::info;
use std::{ffi::OsString, path::Path};
use stft::fft;

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = Args::parse();
    simple_logger::init_with_level(opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = read_wav(&Path::new(&opt.in_file));

    // Bin width is `Fs/N`, `MaxFreq = Fs/2`; where `Fs` is sampling frequency and `N` is samples.
    let bin_width = (spec.sample_rate as f32) / data.len() as f32;
    // Take fft of data.
    let data = fft::scaled_fft(data);

    // Rename input file, change to .png, and append `_out` to basename.
    let file_out = {
        let parent = Path::new(&opt.in_file).parent().unwrap();
        let mut temp = Path::new(&opt.in_file).file_stem().unwrap().to_owned();
        temp.push(OsString::from(format!("_out.png")));
        parent.join(temp)
    };

    info!("Plotting fft with bin width: {bin_width}.");
    plot_fft(
        &opt.in_file,
        file_out.to_str().unwrap(),
        data,
        spec.sample_rate as f32,
        bin_width,
    )?;
    Ok(())
}