use crate::{
    args::{ReceiveOpt, TransceiverOpt, TransmissionCli, TransmitOpt},
    file_io, transmit,
};
use anyhow::Context;
use log::info;
use std::io;

pub fn run(opt: TransmissionCli) -> anyhow::Result<()> {
    // Init logging.
    simple_logger::init_with_level(opt.log_opt.log_level)?;

    match opt.transceiver_opt {
        TransceiverOpt::Transmit(transmit_opt) => transmit_from_file(transmit_opt),
        TransceiverOpt::Receive(receive_opt) => receive_from_file(receive_opt),
    }
}

/// Transmit from file main logic.
pub fn transmit_from_file(opt: TransmitOpt) -> anyhow::Result<()> {
    // Read file bytes.
    let bytes = file_io::read_file_bytes(&opt.in_file.in_file)
        .with_context(|| format!("Opening {}", opt.in_file.in_file.display()))?
        .collect::<Result<Vec<u8>, io::Error>>()
        .context("Reading bytes from file.")?
        .into_iter();

    transmit::encode_transmission(
        opt.fec_spec,
        opt.transmission_spec,
        bytes,
        transmit::play_stream_blocking,
    )
    .map_err(Into::into)
}

/// Receive from file main logic.
pub fn receive_from_file(opt: ReceiveOpt) -> anyhow::Result<()> {
    // Read in wav file.
    let (spec, data) = file_io::read_wav(&opt.in_file.in_file).with_context(|| {
        format!(
            "Opening {} for reading wav contents",
            opt.in_file.in_file.display()
        )
    })?;

    // Decode the file's sound transmission.
    let bytes = transmit::decode_transmission(
        opt.fec_spec,
        opt.transmission_spec,
        data.into_iter().map(|x| x as f32),
        20_000,
        spec.sample_rate as f32,
    )?;

    // Write the decoded byte vector to a file.
    let out_path = &opt.out_file.out_file;
    file_io::write_file_bytes(out_path, &bytes)
        .with_context(|| format!("Writing to file {}.", out_path.display()))?;
    info!("Saved decoded file to {}", opt.out_file.out_file.display());

    Ok(())
}
