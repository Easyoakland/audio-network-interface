#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::IteratorAdapter;

    #[test]
    fn even_integer() {
        assert_eq!(([1, 2].into_iter()).mean(), 1);
        assert_eq!([1, 2, 3, 4].into_iter().mean(), 2);
        assert_eq!([-1, 2, 3, 4].into_iter().mean(), 2);
        assert_eq!([-2, -1, 3, 4].into_iter().mean(), 1);
        assert_eq!([-3, -2, -1, 4].into_iter().mean(), 0);
    }

    #[test]
    fn odd_integer() {
        assert_eq!(([1, 2, 3].into_iter()).mean(), 2);
        assert_eq!([1, 2, 3, 4, 5].into_iter().mean(), 3);
        assert_eq!([-1, 2, 3, 4, 5].into_iter().mean(), 2);
        assert_eq!([-2, -1, 3, 4, 4].into_iter().mean(), 1);
        assert_eq!([-3, -2, -1, 4, 4].into_iter().mean(), 0);
    }

    #[test]
    fn even_float() {
        assert_eq!(([1.0, 2.0].into_iter()).mean(), 1.5);
        assert_eq!([1.0, 2.0, 3.0, 4.0].into_iter().mean(), 2.5);
        assert_eq!([-1.0, 2.0, 3.0, 4.0].into_iter().mean(), 2.0);
        assert_eq!([-2.0, -1.0, 3.0, 4.0].into_iter().mean(), 1.0);
        assert_eq!([-3.0, -2.0, -1.0, 4.0].into_iter().mean(), -0.5);
        // Operations on NAN give more NAN. Ordered Float required for NAN == NAN.
        assert_eq!(
            [
                OrderedFloat(-3.0),
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(f32::NAN),
                OrderedFloat(f32::INFINITY),
                OrderedFloat(f32::NEG_INFINITY),
            ]
            .into_iter()
            .mean(),
            OrderedFloat(f32::NAN)
        );
    }

    #[test]
    fn odd_float() {
        assert_eq!(([1.0, 2.0, 3.0].into_iter()).mean(), 2.0);
        assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0].into_iter().mean(), 3.0);
        assert_eq!([-1.0, 2.0, 3.0, 4.0, 5.0].into_iter().mean(), 2.6);
        assert_eq!([-2.0, -1.0, 3.0, 4.0, 4.0].into_iter().mean(), 1.6);
        assert_eq!([-3.0, -2.0, -1.0, 4.0, 4.0].into_iter().mean(), 0.4);
    }
}
