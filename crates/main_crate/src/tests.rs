use crate::{
    args::FecSpec,
    transmit::{decode_transmission, encode_transmission, DecodingError},
};
use dsp::specs::{FdmSpec, OfdmSpec, TransmissionSpec};
use log::trace;
use proptest::{prelude::ProptestConfig, proptest};
use rand_distr::{Distribution, Normal};
use std::{clone::Clone, io, sync::OnceLock};

/// Used to init logging only once among multiple tests.
static INIT_LOGGING: std::sync::Once = std::sync::Once::new();
// Additive White Gaussian Noise (AWGN)
static NORMAL: OnceLock<Normal<f32>> = std::sync::OnceLock::new();

const FEC_DEFAULT_SPEC: FecSpec = FecSpec { parity_shards: 5 };
const OFDM_DEFAULT_TRANSMISSION_SPEC: TransmissionSpec = TransmissionSpec::Ofdm(OfdmSpec {
    seed: 0,
    short_training_repetitions: 10,
    time_symbol_len: 4800,
    cyclic_prefix_len: 480,
    cross_correlation_threshold: 0.125,
    data_symbols: 32,
    first_bin: 20,
});

// TODO use:
// bytes in proptest::collection::vec(1..u8::MAX,1..(u8::MAX as usize)),
fn simulated_transmit_receive(
    fec_spec: FecSpec,
    transmit_spec: TransmissionSpec,
    length: u8,
    noise_amplitude: f32,
) {
    INIT_LOGGING.call_once(|| {
        simple_logger::init_with_env().unwrap();
    });
    let mut channel = Vec::with_capacity(1);
    let bytes = 0..=length;
    trace!("Transmitting bytes {bytes:?}");

    // Encode to sink.
    encode_transmission(
        fec_spec.clone(),
        transmit_spec.clone(),
        bytes.clone(),
        |x| {
            channel.push(x);
            Ok::<_, io::Error>(()) // error doesn't matter for this test. picked randomly.
        },
    )
    .unwrap();

    // Decode from sink.
    let decoded = match decode_transmission(
        fec_spec,
        transmit_spec,
        channel.remove(0).map(|x| {
            x + noise_amplitude
                * (NORMAL
                    .get_or_init(|| Normal::new(0.0, 1.0).unwrap())
                    .sample(&mut rand::thread_rng()))
        }),
        0,
        48000.0,
    ) {
        Ok(x) => x,
        Err(e) => panic!("{e}"),
    };

    assert_eq!(decoded, bytes.collect::<Vec<_>>());
}

proptest! {
    // Decrease case default from 256 to 10 because these test are slow.
    #![proptest_config(ProptestConfig {cases: 10, max_shrink_iters: 1000, .. ProptestConfig::default()})]

    #[test]
    fn simulated_transmit_fdm_test(
        symbol_time in 1..u64::from(u8::MAX),
        bit_width in 1.0..100.0f32,
        start_freq in 0.0..24000f32,
        parallel_channels in 1..100usize,
    ) {
        let start_freq = (start_freq).min(24000.0 - bit_width * parallel_channels as f32);
        let fdm_spec = FdmSpec {
            symbol_time,
            bit_width,
            start_freq,
            parallel_channels
        };
        encode_transmission(FEC_DEFAULT_SPEC, TransmissionSpec::Fdm(fdm_spec), 0..=255, |x| {
            drop(x);
            Ok::<_, io::Error>(()) // error doesn't matter for this test. picked randomly.
        }).unwrap();
    }

    #[test]
    fn simulated_transmit_receive_ofdm(length: u8) {
        simulated_transmit_receive(FEC_DEFAULT_SPEC, OFDM_DEFAULT_TRANSMISSION_SPEC, length, 0.0);
    }

    #[test]
    #[ignore = "Will fail at some noise level. Random."]
    fn simulated_transmit_receive_noisy_ofdm(length: u8, noise_amplitude in 0.0..=0.5f32) {
        if noise_amplitude.is_finite() {
            simulated_transmit_receive(FEC_DEFAULT_SPEC, OFDM_DEFAULT_TRANSMISSION_SPEC, length, noise_amplitude);
        }
    }
}

#[test]
fn simulated_transmit_ofdm() {
    encode_transmission(
        FEC_DEFAULT_SPEC,
        OFDM_DEFAULT_TRANSMISSION_SPEC,
        0..=255,
        |x| {
            drop(x);
            Ok::<_, io::Error>(()) // error doesn't matter for this test. picked randomly.
        },
    )
    .unwrap();
}

#[test]
fn simulated_fail_to_find_decode_ofdm() {
    match decode_transmission(
        FEC_DEFAULT_SPEC,
        OFDM_DEFAULT_TRANSMISSION_SPEC,
        (0..=255i16).map(f32::from).cycle().take(4800),
        0,
        48000.0,
    ) {
        Err(DecodingError::NoPacket(_)) => (),
        Ok(x) => panic!("Shouldn't be able to decode to: {x:?}"),
        Err(e) => panic!("{e}"),
    }
}
