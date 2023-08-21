use anyhow::Context;
use audio_network_interface::{
    args::LoggingOpt,
    file_io::{read_file_bytes, write_file_bytes},
};
use clap::{CommandFactory, Parser, ValueHint};
use dsp::bit_byte_conversion::bytes_to_bits;
use futures_lite::future::block_on;
use iterator_adapters::IteratorAdapter;
use std::{iter, path::PathBuf};

/// Fix in-place bit mutations using consensus. Does not fix additions or deletions.
#[derive(Debug, Parser)]
struct Opt {
    #[command(flatten)]
    pub log_opt: LoggingOpt,

    /// Output file from consensus
    pub out_file: PathBuf,

    /// Files to get consensus from.
    #[arg(value_hint = ValueHint::FilePath)]
    pub files: Vec<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::parse();
    if opt.files.len() < 3 {
        Opt::command()
            .error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "Need at least 3 input files to form a consensus",
            )
            .exit();
    }
    let mut files = opt
        .files
        .iter()
        .map(|x| block_on(read_file_bytes(x)))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|x| bytes_to_bits(x.map(|x| x.expect("No io errors"))))
        .collect::<Vec<_>>();
    let consensus_file = iter::from_fn(move || {
        let mut symbol_cnts = [0; 3]; // 0 is false, 1 is true, 2 is None
        for i in 0..files.len() {
            match files[i].next() {
                Some(false) => symbol_cnts[0] += 1,
                Some(true) => symbol_cnts[1] += 1,
                None => symbol_cnts[2] += 1,
            };
        }
        symbol_cnts
            .iter()
            .enumerate()
            .max_by_key(|(_i, &x)| x)
            .map(|x| match x.0 {
                0 => Some(false),
                1 => Some(true),
                2 => None,
                _ => unreachable!("len 3 array"),
            })
            .expect("NonEmpty")
    })
    .bits_to_bytes()
    .collect::<Vec<_>>();

    write_file_bytes(&opt.out_file, &consensus_file).context("Writing bytes")?;

    Ok(())
}
