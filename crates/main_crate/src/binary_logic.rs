use crate::{args::TransmissionCli, file_io, transmit};
use anyhow::Context;
use log::info;
use std::{io, path::Path};

/// Transmit from file main logic.
pub fn transmit_from_file(opt: TransmissionCli) -> anyhow::Result<()> {
    // Handle commandline arguments.
    // let opt = TransmissionCli::parse();
    simple_logger::init_with_level(opt.base.log_opt.log_level).unwrap();

    // Read file bytes.
    let bytes = file_io::read_file_bytes(Path::new(&opt.base.file_opt.in_file))
        .with_context(|| format!("Opening {}", opt.base.file_opt.in_file))?
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
pub fn receive_from_file(opt: TransmissionCli) -> anyhow::Result<()> {
    simple_logger::init_with_level(opt.base.log_opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = file_io::read_wav(Path::new(&opt.base.file_opt.in_file));

    // Decode the file's sound transmission.
    let bytes = transmit::decode_transmission(
        opt.fec_spec,
        opt.transmission_spec,
        data.into_iter().map(|x| x as f32),
        20_000,
        spec.sample_rate as f32,
    )?;

    // Write the decoded byte vector to a file.
    let out_path = Path::new(&opt.base.file_opt.out_file);
    file_io::write_file_bytes(out_path, &bytes)
        .with_context(|| format!("Writing to file {:?}.", out_path))?;
    info!("Saved decoded file to {}", opt.base.file_opt.out_file);

    Ok(())
}
