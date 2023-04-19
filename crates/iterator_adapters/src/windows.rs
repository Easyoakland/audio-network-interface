#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct Windows<I: Iterator> {
    iter: I,
    buffer: Vec<I::Item>,
    buffer_len: usize,
    first: bool,
}

impl<I> Windows<I>
where
    I: Iterator,
{
    pub fn new(iter: I, n: usize) -> Self {
        if n == 0 {
            panic!("window size must be non-zero")
        }
        Windows {
            iter,
            buffer: Vec::new(),
            first: true,
            buffer_len: n,
        }
    }
}

impl<I> Iterator for Windows<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        // Initialize buffer on first iteration or return none if not enough items.
        if self.first {
            for _ in 0..self.buffer_len {
                self.buffer.push(self.iter.next()?)
            }
            self.first = false
        } else {
            // Use the next item to replace the oldest item.
            self.buffer.remove(0);
            self.buffer.push(self.iter.next()?);
        }

        Some(self.buffer.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::IteratorAdapter;
    use proptest::proptest;

    proptest! {
        #[test]
        fn windows_same_as_vec(vec: Vec<i32>, window_size in 1usize..255) {
            let iter = vec.clone().into_iter();
            assert_eq!(
                iter.windows(window_size).collect::<Vec<_>>(),
                vec.windows(window_size).collect::<Vec<_>>(),
            );
        }


    }

    #[test]
    #[should_panic]
    fn windows_panic_on_0() {
        Vec::<i32>::new().into_iter().chunks(0).for_each(drop);
        vec![1, 2, 3].into_iter().windows(0).for_each(drop);
    }
}
