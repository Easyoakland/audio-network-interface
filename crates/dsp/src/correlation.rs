use iterator_adapters::IteratorAdapter;
use num_complex::Complex;
use num_traits::{Float, Num, Zero};
use std::{
    clone::Clone,
    iter::Sum,
    ops::{Mul, Neg},
};
use stft::fft::FourierFloat;

/// Represents that there is a sensible representation of self as a `Complex<T>`.
pub trait AsComplex<T> {
    fn as_complex(self) -> Complex<T>;
}

/// Trivial representation for `Complex<T>` as `Complex<T>`.
impl<T> AsComplex<T> for Complex<T> {
    #[inline]
    fn as_complex(self) -> Complex<T> {
        self
    }
}

/// Types with a Zero can be represented as a Complex with only a real nonzero component.
impl<T: Zero> AsComplex<T> for T {
    #[inline]
    fn as_complex(self) -> Complex<T> {
        Complex {
            re: self,
            im: T::zero(),
        }
    }
}

/// Finds pearson coefficient between two series.
/// <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>
pub fn pearson_coefficient<T: FourierFloat>(x: &[T], y: &[T]) -> T {
    let x_mean = x.iter().copied().mean();
    let y_mean = x.iter().copied().mean();
    let n = x.len().max(y.len());
    let mut numerator = T::zero();
    let mut denominator_x_term = T::zero();
    let mut denominator_y_term = T::zero();

    for i in 0..n {
        let x_term = x.get(i).map(|x| *x - x_mean);
        let y_term = y.get(i).map(|x| *x - y_mean);
        let (x_term, y_term) = match (x_term, y_term) {
            (Some(x), Some(y)) => (x, y),
            _ => break,
        };
        numerator = numerator + x_term * y_term;
        denominator_x_term = denominator_x_term + x_term * x_term;
        denominator_y_term = denominator_y_term + y_term * y_term;
    }
    let denominator = denominator_x_term.sqrt() * denominator_y_term.sqrt();
    if denominator == T::zero() {
        T::zero()
    } else {
        numerator / denominator
    }
}

/// Convolves the two signals.
/// <https://en.wikipedia.org/wiki/Convolution>
pub fn convolve<T>(f: &[T], g: &[T], n: usize) -> T
where
    T: Sum + Mul + Sum<<T as Mul>::Output> + Clone,
{
    (0..n.min(f.len()))
        .filter_map(|m| f.get(m).zip(g.get(n - m)))
        .map(|x| (*x.0).clone() * (*x.1).clone())
        .sum()
}

/// Iterator over each value that will be summed to compute cross correlation.
fn cross_correlation_before_sum<'a, T, T2>(
    f: &'a [T2],
    g: &'a [T2],
    n: usize,
    #[allow(non_snake_case)] N: usize,
) -> impl Iterator<Item = Complex<T>> + 'a
where
    T2: AsComplex<T> + Clone,
    T: Clone + Neg<Output = T> + Num,
{
    (0..N) // note this is exclusive upper bound so it matches formula.
        .filter_map(move |m| f.get(m).zip(g.get((m + n) % N)))
        .map(|(f, g)| f.clone().as_complex().conj() * g.clone().as_complex())
}

/// Cross correlates the two signals. Formula is cyclic correlation for finite discrete signal from:
/// <https://en.wikipedia.org/wiki/Cross-correlation>
///
/// A high correlation at n indicates that a feature at f\[m\] appears at g\[n+m\].
pub fn cross_correlation<T, T2>(
    f: &[T2],
    g: &[T2],
    n: usize,
    #[allow(non_snake_case)] N: usize,
) -> Complex<T>
where
    T2: AsComplex<T> + Clone,
    T: Clone + Neg<Output = T> + Num + Sum,
{
    cross_correlation_before_sum(f, g, n, N).sum()
}

/// Correlation scaled for packet onset detection.
/// The pairs of products is divided by the energy as in <https://ieeexplore.ieee.org/document/650240>.
///
/// Example use case is when `g` is a timeshifted version of `f`.
/// Function will output ~1.0 when `f` repeats itself by the timeshift of `g` and ~0.0 where it does not.
///
/// # Arguments
/// - `f`: First series.
/// - `g`: Second series.
/// - `N`: The length of the correlation window.
pub fn cross_correlation_timing_metric_single_value<'a, T, T2>(
    f: &'a [T2],
    g: &'a [T2],
    #[allow(non_snake_case)] N: usize,
) -> T
where
    T2: AsComplex<T> + Clone,
    T: Clone + Neg<Output = T> + Num + Sum + Float,
{
    // The actual formula is |P|^2 / R^2.
    // Where P is the "sum of the pairs of products" (sum m=0 to L-1 of r[d+m].conj * r[d+m+L])
    // and R is the energy (sum m=0 to L-1 of |r[d+m+L]|^2).
    (cross_correlation(f, g, 0, N).norm_sqr())
        / cross_correlation_before_sum(f, g, 0, N) // computes f[i+m] * g[i+m]. If f=g this is norm_sqr
            .map(Complex::norm) // if f and g don't match perfectly then they will still have im component. Use norm to remove. If no im component then this does nothing.
            .sum::<T>()
            .powi(2)
}

/// Correlation scaled for packet onset detection.
/// The pairs of products is divided by the energy as in <https://ieeexplore.ieee.org/document/650240>.
///
/// Example use case is when `g` is a timeshifted version of `f`.
/// Function will output ~1.0 when `f` repeats itself by the timeshift of `g` and ~0.0 where it does not.
///
/// # Arguments
/// - `f`: First series.
/// - `g`: Second series.
/// - `N`: The length of the correlation window.
pub fn cross_correlation_timing_metric<'a, T, T2>(
    f: &'a [T2],
    g: &'a [T2],
    #[allow(non_snake_case)] N: usize,
) -> impl Iterator<Item = T> + 'a
where
    T2: AsComplex<T> + Clone,
    T: Sum + Float,
{
    // The actual formula is |P|^2 / R^2.
    // Where P is the "sum of the pairs of products" (sum m=0 to L-1 of r[d+m].conj * r[d+m+L])
    // and R is the energy (sum m=0 to L-1 of |r[d+m+L]|^2).
    (0..(f.len().min(g.len())))
        .map(move |i| cross_correlation_timing_metric_single_value(&f[i..], &g[i..], N))
}

/// Autocorrelates signal with itself.
/// <https://openofdm.readthedocs.io/en/latest/detection.html>
pub fn auto_correlate<T: FourierFloat + Sum, T2: Clone>(
    samples: &[T2],
    correlation_window_len: usize,
    repeat_len: usize,
) -> impl Iterator<Item = T> + '_
where
    T2: AsComplex<T>,
{
    (0..(samples.len() - correlation_window_len - repeat_len)).map(move |i| {
        (((0..correlation_window_len).map(|j| {
            samples[i + j].clone().as_complex()
                * samples[i + j + repeat_len].clone().as_complex().conj()
        }))
        .sum::<Complex<T>>())
        .norm()
            / ((0..correlation_window_len)
                .map(|j| (samples[i + j]).clone().as_complex().norm_sqr()))
            .sum::<T>()
    })
}
