pub mod mean;
pub mod median;
pub mod order_statistics;
pub mod remove_preceding;

use std::{
    cmp::PartialEq,
    collections::BinaryHeap,
    iter::Peekable,
    ops::{Add, Div},
};

use num_traits::{cast, NumCast, Zero};

impl<T: ?Sized> IteratorAdapter for T where T: Iterator {}
pub trait IteratorAdapter: Iterator {
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

    /// Returns iterator without preceding value.
    fn remove_preceding(self, val: Self::Item) -> Peekable<Self>
    where
        Self: Sized,
        Self::Item: PartialEq,
    {
        let mut iter = self.peekable();
        while let Some(x) = iter.peek() {
            if x != &val {
                break;
            } else {
                iter.next();
            }
        }
        iter
    }
}
