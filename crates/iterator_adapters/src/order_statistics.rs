#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::IteratorAdapter;

    #[test]
    fn even_integer() {
        assert_eq!(([1, 2].into_iter()).kth_order(50), 2);
        assert_eq!([1, 2, 3, 4].into_iter().kth_order(50), 3);
        assert_eq!([-1, 2, 3, 4].into_iter().kth_order(50), 3);
        assert_eq!([-2, -1, 3, 4].into_iter().kth_order(50), 3);
        assert_eq!([-3, -2, -1, 4].into_iter().kth_order(50), -1);
    }

    #[test]
    fn odd_integer() {
        assert_eq!(([1, 2, 3].into_iter()).kth_order(50), 2);
        assert_eq!([1, 2, 3, 4, 5].into_iter().kth_order(50), 3);
        assert_eq!([-1, 2, 3, 4, 5].into_iter().kth_order(50), 3);
        assert_eq!([-2, -1, 3, 4, 4].into_iter().kth_order(50), 3);
        assert_eq!([-3, -2, -1, 4, 4].into_iter().kth_order(50), -1);
    }

    #[test]
    fn even_float() {
        assert_eq!(
            ([OrderedFloat(1.0), OrderedFloat(2.0)].into_iter()).kth_order(50),
            2.0
        );
        assert_eq!(
            [
                OrderedFloat(1.0),
                OrderedFloat(2.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-1.0),
                OrderedFloat(2.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-3.0),
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            -1.0
        );
        assert_eq!(
            [
                OrderedFloat(-3.0),
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(f32::NAN), // Nan is sorted as always greatest by `OrderedFloat`.
                OrderedFloat(f32::INFINITY), // Second greatest
                OrderedFloat(f32::NEG_INFINITY), // Lowest
            ]
            .into_iter()
            .kth_order(50),
            -1.0
        );
    }

    #[test]
    fn odd_float() {
        assert_eq!(
            ([OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)].into_iter()).kth_order(50),
            2.0
        );
        assert_eq!(
            [
                OrderedFloat(1.0),
                OrderedFloat(2.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0),
                OrderedFloat(5.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-1.0),
                OrderedFloat(2.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0),
                OrderedFloat(5.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(3.0),
                OrderedFloat(4.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            3.0
        );
        assert_eq!(
            [
                OrderedFloat(-3.0),
                OrderedFloat(-2.0),
                OrderedFloat(-1.0),
                OrderedFloat(4.0),
                OrderedFloat(4.0)
            ]
            .into_iter()
            .kth_order(50),
            -1.0
        );
    }
}
