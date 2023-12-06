#[cfg(not(target_arch = "wasm32"))]
fn main() {
    panic!("Invalid without wasm")
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use audio_network_interface::binary_logic;
    use klask::Settings;

    klask::run_derived_web(
        {
            let mut settings = Settings::default();
            settings.prefer_long_about = true;
            settings
        },
        |x| async {
            if let Err(e) = binary_logic::run(x).await {
                log::error!("Unhandled exception when attempting to transmit file: {e:?}")
            }
        },
    );
}
