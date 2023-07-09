pub mod bit_to_byte;
pub mod chunks;
pub mod mean;
pub mod median;
pub mod order_statistics;
pub mod remove_preceding;
pub mod windows;

use bit_to_byte::BitToByte;
use chunks::Chunks;
use num_traits::{cast, NumCast, Zero};
use std::{
    cmp::PartialEq,
    collections::BinaryHeap,
    iter::{Map, Peekable},
    ops::{Add, Div},
};
use windows::Windows;

type MappedWindow<I> = Map<Windows<I>, fn(Vec<<I as Iterator>::Item>) -> <I as Iterator>::Item>;

impl<T: ?Sized> IteratorAdapter for T where T: Iterator {}
pub trait IteratorAdapter: Iterator {
    /// Finds the mean of the iterator's values.
    fn mean(self) -> Self::Item
    where
        Self: Sized,
        Self::Item: Zero + NumCast + Div<Self::Item, Output = Self::Item>,
    {
        let (count, sum) = self.fold(
            (0, Zero::zero()),
            |acc: (u32, Self::Item), x: Self::Item| (acc.0 + 1, acc.1 + x),
        );
        sum / cast::<u32, Self::Item>(count)
            .expect("Can't cast iterator count as u32 to Self::Item type.")
    }

    /// Finds the median value of the iterator's items.
    // TODO fix algorithm from O(n log n) to median of medians O(n) selection algorithm.
    fn median(self) -> Self::Item
    where
        Self: Sized,
        Self::Item: Ord
            + Clone
            + Add<Self::Item, Output = Self::Item>
            + Div<Self::Item, Output = Self::Item>
            + NumCast,
    {
        let sorted = self.collect::<BinaryHeap<Self::Item>>().into_sorted_vec();
        // If even median is average of center 2 elements.
        if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1].clone() + sorted[sorted.len() / 2].clone())
                / cast(2).expect("Can't divide by 2")
        }
        // If odd then center is median.
        else {
            sorted[sorted.len() / 2].clone()
        }
    }

    /// Finds the value in the kth percentile from the iterator.
    fn kth_order(self, k: usize) -> Self::Item
    where
        Self: Sized,
        Self::Item: Ord
            + Clone
            + Add<Self::Item, Output = Self::Item>
            + Div<Self::Item, Output = Self::Item>,
    {
        assert!(k <= 100, "k must be < 100");
        let sorted = self.collect::<BinaryHeap<Self::Item>>().into_sorted_vec();
        sorted[(k * sorted.len()) / 100].clone()
    }

    /// Returns iterator without preceding value.
    fn remove_preceding(self, val: Self::Item) -> Peekable<Self>
    where
        Self: Sized,
        Self::Item: PartialEq,
    {
        let mut iter = self.peekable();
        while let Some(x) = iter.peek() {
            if x == &val {
                iter.next();
            } else {
                break;
            }
        }
        iter
    }

    /// Successive window like the implementation for [`Vec`] but for an [`Iterator`].
    /// Returns an iterator over all contiguous windows of length size. The windows overlap.
    fn windows(self, window_size: usize) -> Windows<Self>
    where
        Self: Sized,
    {
        Windows::new(self, window_size)
    }

    /// Finds the mean of each window.
    fn rolling_average(self, window_len: usize) -> MappedWindow<Self>
    where
        Self: Sized,
        Self::Item: Clone,
        Self::Item: Zero + NumCast + Div<Self::Item, Output = Self::Item>,
    {
        self.windows(window_len).map(|x| x.into_iter().mean())
    }

    /// Finds the median of each window.
    fn rolling_median(self, window_len: usize) -> MappedWindow<Self>
    where
        Self: Sized,
        Self::Item: Ord
            + Clone
            + Add<Self::Item, Output = Self::Item>
            + Div<Self::Item, Output = Self::Item>
            + NumCast,
    {
        self.windows(window_len).map(|x| x.into_iter().median())
    }

    /// Successive chunks like the implementation for [`Vec`] but for an [`Iterator`].
    ///
    /// Returns an iterator over all chunks of length `size` plus a possible remainder chunk.
    /// If `size` does not divide the length of the iterator with no remainder, then the last chunk will have length of the remainder.
    ///
    /// The chunks do not overlap.
    fn chunks(self, chunk_size: usize) -> Chunks<Self>
    where
        Self: Sized,
    {
        Chunks::new(self, chunk_size)
    }

    /// Converts iterator of bool into iterator of byte sized type. Extra remainder bits will be ignored.
    fn bits_to_bytes<T>(self) -> BitToByte<Self, T>
    where
        Self: Sized + Iterator<Item = bool>,
    {
        BitToByte::new(self)
    }
}
