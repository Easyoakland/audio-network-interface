//! Encodes and sends the input file through audio.

use audio_network_interface::{args::TransmissionCli, binary_logic};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    // Handle commandline arguments.
    let opt = TransmissionCli::parse();
    futures_lite::future::block_on(binary_logic::run(opt))
}
