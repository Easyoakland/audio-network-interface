//! The function called for each sample of audio must run faster than 1/48000 = 20.833 us. Prefferably much faster.
//! The benchmarks in this file are tests for what can or can not be included in this loop.

use audio_network_interface::transmit;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dsp::{
    carrier_modulation::{bpsk_encode, null_encode, ook_encode},
    ofdm::{OfdmDataEncoder, OfdmFramesEncoder, SubcarrierEncoder},
    specs::OfdmSpec,
};
use std::iter;

fn create_thread(c: &mut Criterion) {
    c.bench_function("create thread", |b| {
        b.iter(|| std::thread::spawn(|| drop(black_box(0))))
    });
}

fn create_vec(c: &mut Criterion) {
    c.bench_function("create vec", |b| {
        b.iter(|| black_box(vec![1000.0f32 + 10.0 * 10.0]))
    });
}

fn create_vec_iter(c: &mut Criterion) {
    c.bench_function("create vec iter", |b| {
        b.iter(|| {
            black_box({
                let vec = vec![1000.0f32 + 10.0 * 10.0];
                let mut vec = vec.into_iter();
                vec.next().unwrap()
            })
        })
    });
}

fn ofdm_benchmark(c: &mut Criterion) {
    let subcarriers = Box::leak(Box::new(
        [SubcarrierEncoder::T0(null_encode::<f32>); stft::time_samples_to_frequency(48)],
    ));
    subcarriers[1] = SubcarrierEncoder::T1(ook_encode);
    let mut ofdm = OfdmDataEncoder::new(
        transmit::bytes_to_bits(iter::repeat(255)),
        subcarriers,
        subcarriers.len() / 10,
    );
    c.bench_function("ofdm benchmark", |b| b.iter(|| black_box(ofdm.next())));
}

fn ofdm_multiframe_benchmark(c: &mut Criterion) {
    let subcarriers = Box::leak(Box::new(
        [SubcarrierEncoder::T0(null_encode::<f32>); stft::time_samples_to_frequency(48)],
    ));
    subcarriers[1] = SubcarrierEncoder::T1(bpsk_encode);
    let ofdm_spec = OfdmSpec {
        seed: Default::default(),
        short_training_repetitions: 10,
        time_symbol_len: 48,
        cyclic_prefix_len: 10,
        cross_correlation_threshold: 0.12,
        data_symbols: 32,
        first_bin: 20,
    };
    let mut ofdm =
        OfdmFramesEncoder::new([false, true].into_iter().cycle(), subcarriers, ofdm_spec).flatten();
    c.bench_function("ofdm multiframe benchmark", |b| {
        b.iter(|| black_box(ofdm.next()))
    });
}

criterion_group!(
    benches,
    create_thread,
    create_vec,
    create_vec_iter,
    ofdm_benchmark,
    ofdm_multiframe_benchmark
);
criterion_main!(benches);
