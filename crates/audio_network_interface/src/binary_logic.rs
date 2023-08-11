use crate::{
    args::{ReceiveOpt, TransceiverOpt, TransmissionCli, TransmitOpt},
    constants::SKIPPED_STARTUP_SAMPLES,
    file_io, transmit,
};
use anyhow::Context;
use log::info;
#[cfg(target_arch = "wasm32")]
use log::{Level, LevelFilter};
use std::io;

pub async fn run(opt: TransmissionCli) -> anyhow::Result<()> {
    // Init logging.
    #[cfg(not(target_arch = "wasm32"))]
    simple_logger::init_with_level(opt.log_opt.log_level)?;
    #[cfg(target_arch = "wasm32")]
    klask::logger::Logger::set_max_level(match opt.log_opt.log_level {
        Level::Error => LevelFilter::Error,
        Level::Warn => LevelFilter::Warn,
        Level::Info => LevelFilter::Info,
        Level::Debug => LevelFilter::Debug,
        Level::Trace => LevelFilter::Trace,
    });

    match opt.transceiver_opt {
        TransceiverOpt::Transmit(transmit_opt) => transmit_from_file(transmit_opt).await,
        TransceiverOpt::Receive(receive_opt) => receive_from_file(receive_opt).await,
    }
}

/// Transmit from file main logic.
pub async fn transmit_from_file(opt: TransmitOpt) -> anyhow::Result<()> {
    // Read file bytes.
    let bytes = file_io::read_file_bytes(&opt.in_file.in_file)
        .await
        .with_context(|| format!("Opening file {}", opt.in_file.in_file.display()))?
        .collect::<Result<Vec<u8>, io::Error>>()
        .context("Reading bytes from file.")?
        .into_iter();

    transmit::encode_transmission(
        opt.fec_spec,
        opt.transmission_spec,
        bytes,
        transmit::play_stream,
    )
    .await
    .context("error building stream")?
    .map_err(Into::into)
}

/// Receive from file main logic.
pub async fn receive_from_file(opt: ReceiveOpt) -> anyhow::Result<()> {
    // Read in wav file.
    let (spec, data) = file_io::read_wav(&opt.in_file.in_file)
        .await
        .with_context(|| {
            format!(
                "Opening {} for reading wav contents",
                opt.in_file.in_file.display()
            )
        })?;

    // Decode the file's sound transmission.
    // TODO `skip while sample == 0.0` instead of hardcoded skipped samples.
    let bytes = transmit::decode_transmission(
        opt.fec_spec,
        opt.transmission_spec,
        data.into_iter().map(|x| x as f32),
        SKIPPED_STARTUP_SAMPLES,
        spec.sample_rate as f32,
    )?;

    // Write the decoded byte vector to a file.
    let out_path = &opt.out_file.out_file;
    file_io::write_file_bytes(out_path, &bytes)
        .with_context(|| format!("Writing to file {}.", out_path.display()))?;
    info!(
        "Saved decoded file to '{}'",
        opt.out_file.out_file.display()
    );

    Ok(())
}
