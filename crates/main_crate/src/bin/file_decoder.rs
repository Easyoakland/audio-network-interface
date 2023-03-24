//! Decodes the audio in a .wav file into the corresponding data that was transmitted.
//! Useful as a repeatable dry-run of the logic used in the live receiver binary.

use audio_network_interface::{
    args::transmit_receive_args::Args,
    file_io::{read_wav, write_file_bytes},
    receive::AmplitudeModulationDecoder,
    transmit::bytes_to_bits,
};
use bitvec::vec::BitVec;
use clap::Parser as _;
use fec::{parity, reed_solomon};
use log::info;
use std::{io::Read, path::Path, time::Duration};
use stft::{fft::window_fn, SpecCompute, WindowLength};

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = Args::parse();
    simple_logger::init_with_level(opt.log_level).unwrap();

    // Read in wav file.
    let (spec, data) = read_wav(Path::new(&opt.in_file));

    // Compute stft of data.
    let window_len = WindowLength::from_duration(
        Duration::from_millis(opt.symbol_time),
        spec.sample_rate as f32,
    );
    let spec_compute = SpecCompute::new(data, window_len, window_len / 1, window_fn::hann);
    let stft = spec_compute.stft();

    // Setup transmission parameters.
    let frequency_channels = (0..opt.parallel_channels)
        .map(|i| opt.start_freq + (1 + i) as f32 * opt.bit_width)
        .collect();
    let decoder = AmplitudeModulationDecoder {
        bits_cnt: 256 * 8 * 10,
        frequency_channels,
        sample_rate: spec.sample_rate as f32,
    };

    // Decode the transmission.
    // TODO dynamically change sensitivity until beginning and end are detected and are at different positions.
    let bytes = decoder.decode_amplitude_modulation(&stft, 1.0);

    log::debug!("Bytes_before_decoding:\n{:?}", &bytes);

    // Convert to bits to do shard recovery.
    let bits = bytes_to_bits(bytes.into_iter());

    // Dinterlace and remove invalid shards.
    let shards = {
        let mut interleaved = BitVec::new();
        interleaved.extend(bits);
        let (data, parity) = parity::deinterleave(&interleaved, 32);
        log::debug!(
            "Data deinterleaved:\n{:?}",
            data.clone() /* .bytes()
                         .map(Result::unwrap)
                         .collect::<Vec<_>>()*/
        );
        let parity_errors = parity::validate_parity(&data, &parity, 32);

        log::debug!(
            "Shard before parity: {:?}",
            data.chunks(32)
                // .map(|shard| Option::Some(shard.bytes().map(Result::unwrap).collect::<Vec<_>>()))
                .last()
                // .unwrap()
                .unwrap() // .collect::<Vec<_>>()
        );

        data.chunks(32)
            .zip(parity_errors)
            .map(|(shard, invalid)| {
                // TODO check first shard is number followed by that many zeros.
                // TODO if shard padding is > shard size then fail it.
                if invalid || shard.len() != 32 {
                    None
                } else {
                    Some(shard.bytes().map(Result::unwrap).collect::<Vec<_>>())
                }
            })
            .collect::<Vec<_>>()
    };

    info!(
        "Parity failed shards: {}",
        shards
            .iter()
            .flat_map(|x| if x.is_none() { Some(0) } else { None })
            .count(),
    );
    log::debug!("Shards after parity check:\n{:?}", shards);

    // Attempt to reconstruct data from remaining shards.
    let bytes = reed_solomon::reconstruct_data(shards, 5)?;

    log::debug!("Shards after reconstruction:\n{:?}", bytes);

    // Write the decoded byte vector to a file.
    write_file_bytes(Path::new(&opt.out_file), &bytes)?;
    info!("Saved decoded file to {}", opt.out_file);
    Ok(())
}
