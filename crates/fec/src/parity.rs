use bitvec::{order::BitOrder, slice::BitSlice, store::BitStore, vec::BitVec};
use std::{io::Read, iter};

use crate::traits::Function;

pub type BitStream = BitVec<u8, bitvec::prelude::LocalBits>;

/// Encodes and decodes data by interleaving parity bits.
#[derive(Debug, Default, Clone)]
pub struct ParityEncoder {
    pub data_block_size: usize,
}

impl ParityEncoder {
    /// Generates a parity check bit for every `data_block_size` block of bits.
    pub fn generate_parity<T: BitStore, O: BitOrder>(&self, data: &BitSlice<T, O>) -> BitVec<T, O> {
        let mut parity = BitVec::with_capacity(data.len() / self.data_block_size);
        for block in data.chunks(self.data_block_size) {
            // Make block even.
            if block
                .into_iter()
                .map(|x| if *x { 1 } else { 0 })
                .sum::<u8>()
                % 2
                == 0
            {
                parity.push(false)
            } else {
                parity.push(true)
            };
        }
        parity
    }

    /// Interleave parity bits to the end of the block they parity.
    #[must_use = "Doesn't affect inputs."]
    pub fn interleave<O: BitOrder, T: BitStore>(
        &self,
        data: &BitSlice<T, O>,
        parity: &BitSlice<T, O>,
    ) -> BitVec<T, O> {
        // Interleave parity to end of data block
        let mut out = BitVec::new();
        let data_iter = data.chunks(self.data_block_size);
        let mut parity_iter = parity.iter();
        for chunk in data_iter {
            out.extend(chunk.iter().chain(iter::once(
                parity_iter.next().expect("Insufficient parity bits"),
            )));
        }
        out
    }
}

impl Function for ParityEncoder {
    type Input = BitStream;
    type Output = BitStream;
    fn map(&self, source: Self::Input) -> Self::Output {
        let parity = self.generate_parity(&source);
        self.interleave(&source, &parity)
    }
}

/// Decodes and verifies data by de-interleaving parity bits.
/// Input is `BitVec`. Output is `Vec<Option<Vec<u8>>>`;
#[derive(Debug, Default, Clone)]
pub struct ParityDecoder {
    pub data_block_size: usize,
}

impl ParityDecoder {
    /// Compares parity bits to parity bits expected from data.
    /// Returns a bitvector with ones at each invalid block.
    pub fn validate_parity<T: BitStore, O: BitOrder>(
        &self,
        data: &BitSlice<T, O>,
        parity: &BitSlice<T, O>,
    ) -> BitVec<T, O> {
        let mut validated = BitVec::with_capacity(data.len() / self.data_block_size);
        for (block_idx, block) in data.chunks(self.data_block_size).enumerate() {
            // Check block even.
            // Mark invalid blocks with true (1). Valid blocks with false (0).
            if block
                .into_iter()
                .map(|x| if *x { 1 } else { 0 })
                .sum::<u8>()
                % 2
                == 0
            {
                validated.push(parity[block_idx]);
            } else {
                validated.push(!parity[block_idx]);
            };
        }
        validated
    }

    /// De-interleave parity bits into separate (data, parity).
    #[must_use = "Doesn't affect inputs."]
    pub fn deinterleave<T: BitStore, O: BitOrder>(
        &self,
        interleaved: &BitSlice<T, O>,
    ) -> (BitVec<T, O>, BitVec<T, O>) {
        let mut parity = BitVec::new();
        let mut data = BitVec::new();

        // Block_size is number of data bits. +1 for parity bit.
        for block_with_parity in interleaved.chunks(self.data_block_size + 1) {
            // Use each block's individual length to handle remainder block at the end that isn't necessarily block_size.
            let (data_bits, parity_bit) = block_with_parity.split_at(block_with_parity.len() - 1);
            data.extend(data_bits);
            parity.extend(parity_bit);
        }

        (data, parity)
    }
    pub fn decode(&self, input: BitStream) -> Vec<Option<Vec<u8>>> {
        let (data, parity) = self.deinterleave(&input);
        let parity_errors = self.validate_parity(&data, &parity);

        let out = data
            .chunks(self.data_block_size)
            .zip(parity_errors)
            .map(|(shard, invalid)| {
                // TODO check first shard is number followed by that many zeros.
                // TODO if shard padding is > shard size then fail it.
                if invalid || shard.len() != self.data_block_size {
                    None
                } else {
                    Some(shard.bytes().map(Result::unwrap).collect::<Vec<_>>())
                }
            })
            .collect::<Vec<_>>();
        log::trace!(
            "{}/{} failed parity checks at indices: {:?}",
            out.clone().into_iter().filter(|x| x.is_none()).count(),
            out.len(),
            out.clone()
                .into_iter()
                .enumerate()
                .filter(|(_, x)| x.is_none())
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        );
        out
    }
}

impl Function for ParityDecoder {
    type Input = BitStream;

    type Output = Vec<Option<Vec<u8>>>;

    fn map(&self, input: Self::Input) -> Self::Output {
        self.decode(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::Lsb0;

    #[test]
    fn test_generate_parity() {
        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            for bit in [
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
                false, false, // 0
            ] {
                data.push(bit);
            }
            data
        };
        let coder = ParityEncoder { data_block_size: 5 };
        let parity = coder.generate_parity(&data);
        assert_eq!(parity.len(), 3);
        assert!(parity[0]);
        assert!(!parity[1]);
        assert!(!parity[2]);

        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            for bit in [
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
            ] {
                data.push(bit);
            }
            data
        };
        let parity = coder.generate_parity(&data);
        assert_eq!(parity.len(), 2);
        assert!(parity[0]);
        assert!(!parity[1]);
    }

    #[test]
    fn test_validate_parity() {
        let mut data: BitVec<u8, Lsb0> = {
            let mut data = BitVec::new();
            for bit in [
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
                false, false, // 0
            ] {
                data.push(bit);
            }
            data
        };
        let encoder = ParityEncoder { data_block_size: 5 };
        let decoder = ParityDecoder { data_block_size: 5 };
        let parity = encoder.generate_parity(&data);
        let parity_errors = decoder.validate_parity(&data, &parity);
        for (i, bit) in parity_errors.into_iter().enumerate() {
            assert!(!bit, "Failed on iteration {i}");
        }
        *data.get_mut(0).unwrap() = !data[0];
        let parity_errors = decoder.validate_parity(&data, &parity);
        assert_eq!(parity_errors.len(), 3);
        assert!(parity_errors[0]);
        assert!(!parity_errors[1]);

        let mut data = {
            let mut data: BitVec<u8, Lsb0> = BitVec::new();
            for bit in [
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
            ] {
                data.push(bit);
            }
            data
        };
        let parity = encoder.generate_parity(&data);
        let parity_errors = decoder.validate_parity(&data, &parity);
        assert_eq!(parity_errors.len(), 2);
        for (i, bit) in parity_errors.into_iter().enumerate() {
            assert!(!bit, "Failed on iteration {i}");
        }
        *data.get_mut(0).unwrap() = !data[0];
        let parity_errors = decoder.validate_parity(&data, &parity);
        assert!(parity_errors[0]);
    }

    #[test]
    fn test_parity_interleave() {
        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            data.extend([
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
                false, false, // 0
            ]);
            data
        };
        let encoder = ParityEncoder { data_block_size: 5 };
        let parity = encoder.generate_parity(&data);
        let interleaved = encoder.interleave(&data, &parity);
        assert_eq!(interleaved, {
            let mut out: BitVec<usize, Lsb0> = BitVec::new();
            out.extend([
                false, true, false, true, true, true, // 3 + 1
                true, true, false, false, false, false, // 2 + 0
                false, false, false, // 0 + 0
            ]);
            out
        });

        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            data.extend([
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
            ]);
            data
        };
        let parity = encoder.generate_parity(&data);
        let interleaved = encoder.interleave(&data, &parity);
        assert_eq!(interleaved, {
            let mut out: BitVec<usize, Lsb0> = BitVec::new();
            out.extend([
                false, true, false, true, true, true, // 3 + 1
                true, true, false, false, false, false, // 2 + 0
            ]);
            out
        });
    }

    #[test]
    fn test_parity_deinterleave() {
        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            data.extend([
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
                false, false, // 0
            ]);
            data
        };
        let encoder = ParityEncoder { data_block_size: 5 };
        let decoder = ParityDecoder { data_block_size: 5 };
        let parity = encoder.generate_parity(&data);
        let interleaved = encoder.interleave(&data, &parity);
        let deinterleaved = decoder.deinterleave(&interleaved);
        assert_eq!(deinterleaved.0, data);
        assert_eq!(deinterleaved.1, parity);

        let data = {
            let mut data: BitVec<usize, Lsb0> = BitVec::new();
            data.extend([
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
            ]);
            data
        };
        let parity = encoder.generate_parity(&data);
        let interleaved = encoder.interleave(&data, &parity);
        let deinterleaved = decoder.deinterleave(&interleaved);
        assert_eq!(deinterleaved.0, data);
        assert_eq!(deinterleaved.1, parity);
    }
}
