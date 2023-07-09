use bitvec::{
    prelude::{BitArray, LocalBits, Lsb0},
    store::BitStore,
};
use std::iter;

/// Coverts byte to iterator of bool.
pub fn byte_to_bits<A: BitStore>(byte: A) -> bitvec::array::IntoIter<A, LocalBits> {
    BitArray::<A, Lsb0>::from(byte).into_iter()
}

/// Return type of `bytes_to_bits`. Basically `impl Iterator<Item = bool>`.
type BytesToBitIter<A, T> =
    iter::FlatMap<T, bitvec::array::IntoIter<A, Lsb0>, fn(A) -> bitvec::array::IntoIter<A, Lsb0>>;

/// Converts an iterator of bytes into a 8x longer iterator of bits.
/// The returned type is `impl Iterator<Item = bool>` but keeps other traits (ex. `Clone`) the original iterator has.
pub fn bytes_to_bits<A: BitStore, T: Iterator<Item = A>>(bytes: T) -> BytesToBitIter<A, T> {
    bytes.into_iter().flat_map(byte_to_bits)
}
