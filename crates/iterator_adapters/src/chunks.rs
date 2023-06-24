#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct Chunks<I: Iterator> {
    iter: I,
    buffer: Vec<I::Item>,
    buffer_len: usize,
}

impl<I> Chunks<I>
where
    I: Iterator,
{
    pub fn new(iter: I, n: usize) -> Self {
        assert!(n != 0, "chunk size must be non-zero");
        Chunks {
            iter,
            buffer: Vec::new(),
            buffer_len: n,
        }
    }
}

impl<I> Iterator for Chunks<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        // Clear previous chunk.
        self.buffer.clear();
        // Add a chunk (or what remains) to the buffer.
        for _ in 0..self.buffer_len {
            self.buffer.push(match self.iter.next() {
                Some(x) => x,
                None => break,
            });
        }

        if self.buffer.is_empty() {
            return None;
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
        fn chunk_same_as_vec(vec: Vec<i32>, chunk_size in 1usize..255) {
            let iter = vec.clone().into_iter();
            assert_eq!(
                iter.chunks(chunk_size).collect::<Vec<_>>(),
                vec.chunks(chunk_size).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    #[should_panic]
    fn chunks_panic_on_0() {
        Vec::<i32>::new().into_iter().chunks(0).for_each(drop);
        vec![1, 2, 3].into_iter().chunks(0).for_each(drop);
    }
}
