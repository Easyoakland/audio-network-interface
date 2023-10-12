use std::{
    marker::PhantomData,
    ops::{BitOrAssign, Shl},
};

pub struct BitToByte<I, T> {
    iter: I,
    byte_type: PhantomData<T>,
}

impl<I, T> BitToByte<I, T> {
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            byte_type: PhantomData,
        }
    }
}

impl<I, T> Iterator for BitToByte<I, T>
where
    I: Iterator<Item = bool>,
    T: From<bool> + Shl<u32> + Default + BitOrAssign<<T as Shl<u32>>::Output>,
{
    type Item = Result<T, (u32, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_byte = T::default();
        for i in 0..(u8::BITS
            * u32::try_from(std::mem::size_of::<T>())
                .expect("T bytes length is too large for a u32"))
        {
            let bit = match (i, self.iter.next()) {
                (0, None) => return None, // If on the first bit there is no new value
                (_, None) => return Some(Err((i, next_byte))), // If partially filled
                (_, Some(x)) => x,        // If still more bits left
            };
            next_byte |= T::from(bit) << i;
        }
        Some(Ok(next_byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_byte_u8() {
        let bytes = [1, 0, 1, 0, 1, 0, 1, 0]
            .repeat(2)
            .into_iter()
            .rev()
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::<_, u8>::new(bytes);
        assert_eq!(bytes_iter.next(), Some(Ok(0b10101010)));
        assert_eq!(bytes_iter.next(), Some(Ok(0b10101010)));
        assert_eq!(bytes_iter.next(), None);
    }

    #[test]
    fn full_byte_u32() {
        let bytes = [1, 0, 1, 0, 1, 0, 1, 0]
            .repeat(4)
            .into_iter()
            .rev()
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::<_, u32>::new(bytes);
        assert_eq!(
            bytes_iter.next(),
            Some(Ok(0b10101010_10101010_10101010_10101010))
        );
        assert_eq!(bytes_iter.next(), None);
    }

    #[test]
    fn partial_byte_u8() {
        let bytes = [1, 1, 0, 0, 1, 1, 0, 1]
            .into_iter()
            .chain([1, 0, 1, 0])
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::<_, u8>::new(bytes);
        assert_eq!(bytes_iter.next(), Some(Ok(0b10110011)));
        assert_eq!(bytes_iter.next(), Some(Err((4, 0b0101))));
        assert_eq!(bytes_iter.next(), None);
    }

    #[test]
    fn partial_byte_u32() {
        let bytes = [1, 0, 1, 0, 1, 1, 0, 0]
            .repeat(4)
            .into_iter()
            .chain([0, 1, 1, 1])
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::<_, u32>::new(bytes);
        assert_eq!(
            bytes_iter.next(),
            Some(Ok(0b00110101_00110101_00110101_00110101))
        );
        assert_eq!(bytes_iter.next(), Some(Err((4, 0b1110))));
        assert_eq!(bytes_iter.next(), None);
    }
}
