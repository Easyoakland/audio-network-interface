//! This example will compare the input file to the output file to determine
//! which bits were successfully transmitted from the input file to the output file.
//! It will show how often each bit place was wrong.

use audio_network_interface::{args::BaseCli, file_io::read_file_bytes};
use clap::Parser as _;
use fixedbitset::FixedBitSet;
use std::path::Path;

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = BaseCli::parse();
    simple_logger::init_with_level(opt.log_opt.log_level).unwrap();

    // Read in files.
    let input = read_file_bytes(Path::new(&opt.file_opt.in_file))?;
    let output = read_file_bytes(Path::new(&opt.file_opt.out_file))?;

    // Count number of times each bit is different.
    let mut errors = [0u32; 8];
    let mut file_len_in_bytes = 0u32;
    for (in_byte, out_byte) in input.zip(output) {
        file_len_in_bytes += 1;
        let in_byte = FixedBitSet::with_capacity_and_blocks(8, vec![in_byte?.into()]);
        let out_byte = FixedBitSet::with_capacity_and_blocks(8, vec![out_byte?.into()]);
        for dif_bit in in_byte.symmetric_difference(&out_byte) {
            errors[dif_bit] += 1;
        }
    }

    // Output results.
    println!(
        "Errors percentages per bit (1 up to 7): {:?}",
        errors
            .iter()
            .map(|&x| x as f32 / file_len_in_bytes as f32)
            .map(|x| x * 100.0)
            .collect::<Vec<_>>()
    );
    println!(
        "Total error percentage: {}",
        100.0 * errors.iter().sum::<u32>() as f32 / (file_len_in_bytes as f32 * 8.0)
    );

    Ok(())
}
