//! Encodes and sends the input file through audio. Uses native gui.
#![windows_subsystem = "windows"]

use audio_network_interface::binary_logic;
use klask::Settings;

fn main() {
    klask::run_derived_native(Settings::default(), |opt| {
        futures_lite::future::block_on(binary_logic::run(opt)).unwrap();
    });
}
