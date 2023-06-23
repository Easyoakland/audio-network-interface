//! Encodes and sends the input file through audio. Uses native gui.

use audio_network_interface::binary_logic;
use klask::Settings;

fn main() {
    klask::run_derived_native(Settings::default(), |opt| {
        binary_logic::run(opt).unwrap();
    });
}
