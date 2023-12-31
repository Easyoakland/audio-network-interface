//! Encodes and sends the input file through audio. Uses native gui.
#![windows_subsystem = "windows"]

use audio_network_interface::binary_logic;
use klask::Settings;

fn main() {
    klask::run_derived_native(
        {
            let mut settings = Settings::default();
            settings.prefer_long_about = true;
            settings
        },
        |opt| {
            if let Err(e) = futures_lite::future::block_on(binary_logic::run(opt)) {
                log::error!("Unhandled exception when attempting to transmit file: {e:?}");
            }
        },
    );
}
