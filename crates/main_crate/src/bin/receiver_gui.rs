//! Decodes the audio in a .wav file into the corresponding data that was transmitted.
//! Useful as a repeatable dry-run of the logic used in the live receiver binary.

use audio_network_interface::binary_logic;
use klask::Settings;

fn main() {
    klask::run_derived(Settings::default(), |opt| {
        binary_logic::receive_from_file(opt).unwrap();
    });
}
