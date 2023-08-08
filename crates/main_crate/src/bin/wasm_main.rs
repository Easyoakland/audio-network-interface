#[cfg(not(target_arch = "wasm32"))]
fn main() {
    panic!("Invalid without wasm")
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use audio_network_interface::binary_logic;
    use klask::Settings;

    klask::run_derived_web(Settings::default(), |x| async {
        binary_logic::run(x)
            .await
            .expect("Unhandled exception when attempting to transmit file")
    });
    log::info!("test");
}
