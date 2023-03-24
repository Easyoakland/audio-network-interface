#[cfg(test)]
mod tests {
    use crate::IteratorAdapter;

    #[test]
    fn preceded_test() {
        assert_eq!(
            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0]
                .into_iter()
                .remove_preceding(0)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 0, 0]
        );

        assert_eq!(
            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0]
                .into_iter()
                .remove_preceding(0)
                .rev()
                .remove_preceding(0)
                .rev()
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }
}
