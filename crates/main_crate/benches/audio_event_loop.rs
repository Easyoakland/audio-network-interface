//! The function called for each sample of audio must run faster than 1/48000 = 20.833 us. Prefferably much faster.
//! The benchmarks in this file are tests for what can or can not be included in this loop.
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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
criterion_group!(benches, create_thread, create_vec, create_vec_iter);
criterion_main!(benches);
