//! Encodes and sends the input file through audio.

use std::{io, path::Path, time::Duration};

use audio_network_interface::{
    args::transmit_receive_args::Args,
    file_io,
    transmit::{self, AmplitudeModulationConfig},
};
use bitvec::{prelude::Lsb0, vec::BitVec};
use clap::Parser as _;
use cpal::traits::{DeviceTrait as _, HostTrait as _};
use fec::{parity, reed_solomon};
use log::{info, trace};
use stft::WindowLength;

fn main() -> anyhow::Result<()> {
    // Handle commandline arguments.
    let opt = Args::parse();
    simple_logger::init_with_level(opt.log_level).unwrap();

    // Read file bytes.
    let bytes: Result<Vec<u8>, io::Error> =
        file_io::read_file_bytes(Path::new(&opt.in_file))?.collect();
    let bytes = bytes?;

    // Encode with reed_solomon.
    let bytes = reed_solomon::encode(bytes, 4, 5)?;

    log::debug!("Bytes transmitting: {:?}", &bytes);

    // Add parity checks to shards and convert bytes to bits.
    let mut bits: BitVec<u8, Lsb0> = BitVec::new();
    for shard in bytes {
        let shard_bitvec = &BitVec::<_, Lsb0>::from_slice(&shard);
        let parity = parity::generate_parity(&shard_bitvec, 8 * shard.len());
        bits.append(&mut parity::interleave(
            &shard_bitvec,
            &parity,
            8 * shard.len(),
        ));
    }
    let bits = bits.into_iter();

    // Get speaker sample_rate.
    let sample_rate = cpal::default_host()
        .default_output_device()
        .expect("no output device available")
        .default_output_config()
        .unwrap()
        .sample_rate()
        .0;

    // Generate speaker amplitude iterator.
    // TODO replace leaking with scoped thread.
    // Not a big deal because config only needs to be made once to use for all future signals.
    let config = Box::leak(Box::new(AmplitudeModulationConfig::new(
        opt.bit_width,
        opt.parallel_channels,
        sample_rate as f32,
        opt.start_freq,
        WindowLength::from_duration(Duration::from_millis(opt.symbol_time), sample_rate as f32)
            .samples(),
    )));
    let encoded = config.encode(bits);

    // Get duration of full tranmission.
    let transmit_length = encoded.clone().count();
    let duration_of_transmission =
        Duration::from_secs_f32(transmit_length as f32 / sample_rate as f32);

    // Play tranmission.
    let _stream = transmit::play_freq(encoded)?;
    info!("Transmission of length {duration_of_transmission:?} playing.");

    // Stream will play until it is dropped.
    std::thread::sleep(duration_of_transmission);
    trace!("Dropping played stream");

    Ok(())
}
