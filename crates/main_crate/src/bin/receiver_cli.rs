//! Decodes the audio in a .wav file into the corresponding data that was transmitted.
//! Useful as a repeatable dry-run of the logic used in the live receiver binary.

use audio_network_interface::{args::TransmissionCli, binary_logic};
use clap::Parser;

fn main() -> Result<(), anyhow::Error> {
    // Handle commandline arguments.
    let opt = TransmissionCli::parse();
    binary_logic::receive_from_file(opt)
}
