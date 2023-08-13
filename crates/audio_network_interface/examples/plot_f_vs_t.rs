//! Plots the frequencies corresponding the data channels over time.
//! Useful for analyzing why a bit is misinterpreted on the receiver.

use audio_network_interface::{args::BaseCli, file_io::read_wav, plotting::plot_series};
use clap::{command, Parser};
use dsp::specs::FdmSpec;
use futures_lite::future::block_on;
use stft::{fft::window_fn, SpecCompute, WindowLength};

#[derive(Parser)]
struct Opt {
    #[command(flatten)]
    pub base: BaseCli,

    #[command(flatten)]
    pub fdm_spec: FdmSpec,
}

fn main() -> Result<(), anyhow::Error> {
    block_on(async_main())
}

async fn async_main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = Opt::parse();
    simple_logger::init_with_level(opt.base.log_opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = read_wav(&opt.base.file_in.in_file).await?;

    // Set window parameters so each symbol is one sample.
    let window_len = WindowLength::from_duration(
        std::time::Duration::from_millis(opt.fdm_spec.symbol_time),
        spec.sample_rate as f32,
    );
    let window_step = window_len;

    // Compute stft of data.
    let spec_compute = SpecCompute::new(data, window_len, window_step, window_fn::hann);
    let frequency_analysis = spec_compute.stft();

    for f in (1..=opt.fdm_spec.parallel_channels)
        .map(|i| (opt.fdm_spec.start_freq + i as f32 * opt.fdm_spec.bit_width))
    {
        let transient = frequency_analysis
            .get_bin(f, spec.sample_rate as f32)
            .unwrap();

        // Rename input file, change to .png, and append `_out` to basename.
        let file_out = {
            let parent = opt.base.file_in.in_file.parent().unwrap();
            let mut temp = opt.base.file_in.in_file.file_stem().unwrap().to_owned();
            temp.push(format!("_time_vs_channel_{f}.png"));
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

    Ok(())
}
