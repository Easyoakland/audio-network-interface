use num_traits::Zero;
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
    T: From<bool> + Shl<u32> + Zero + BitOrAssign<<T as Shl<u32>>::Output>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_byte = T::zero();
        for i in 0..(u8::BITS
            * u32::try_from(std::mem::size_of::<T>())
                .expect("T bytes length is too large for a u32"))
        {
            let bit = self.iter.next()?;
            next_byte |= T::from(bit) << i;
        }
        Some(next_byte)
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
        let mut bytes_iter = BitToByte::new(bytes);
        assert_eq!(bytes_iter.next().unwrap(), 0b10101010);
        assert_eq!(bytes_iter.next().unwrap(), 0b10101010);
        assert_eq!(bytes_iter.next(), None::<u8>);
    }

    #[test]
    fn full_byte_u32() {
        let bytes = [1, 0, 1, 0, 1, 0, 1, 0]
            .repeat(4)
            .into_iter()
            .rev()
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::new(bytes);
        assert_eq!(
            bytes_iter.next().unwrap(),
            0b10101010101010101010101010101010
        );
        assert_eq!(bytes_iter.next(), None::<u32>);
    }

    #[test]
    fn partial_byte_u8() {
        let bytes = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            .into_iter()
            .rev()
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::new(bytes);
        assert_eq!(bytes_iter.next().unwrap(), 0b10101010);
        assert_eq!(bytes_iter.next(), None::<u8>);
    }

    #[test]
    fn partial_byte_u32() {
        let bytes = [1, 0, 1, 0, 1, 0, 1, 0]
            .repeat(4)
            .into_iter()
            .chain([1, 0, 1, 0])
            .rev()
            .map(|x| x != 0);
        let mut bytes_iter = BitToByte::new(bytes);
        assert_eq!(
            bytes_iter.next().unwrap(),
            0b10101010101010101010101010101010
        );
        assert_eq!(bytes_iter.next(), None::<u32>);
    }
}
