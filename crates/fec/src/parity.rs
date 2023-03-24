use bitvec::{order::BitOrder, slice::BitSlice, store::BitStore, vec::BitVec};
use std::iter;

/// Generates a parity check bit for every `data_block_size` block of bits.
pub fn generate_parity<T: BitStore, O: BitOrder>(
    data: &BitSlice<T, O>,
    data_block_size: usize,
) -> BitVec<T, O> {
    let mut parity = BitVec::with_capacity(data.len() / data_block_size);
    for block in data.chunks(data_block_size) {
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

/// Compares parity bits to parity bits expected from data.
/// Returns a bitvector with ones at each invalid block.
pub fn validate_parity(data: &BitSlice, parity: &BitSlice, data_block_size: usize) -> BitVec {
    let mut validated = BitVec::with_capacity(data.len() / data_block_size);
    for (block_idx, block) in data.chunks(data_block_size).enumerate() {
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

/// Interleave parity bits to the end of the block they parity.
#[must_use = "Doesn't affect inputs."]
pub fn interleave<O: BitOrder, T: bitvec::store::BitStore>(
    data: &BitSlice<T, O>,
    parity: &BitSlice<T, O>,
    data_block_size: usize,
) -> BitVec {
    // Interleave parity to end of data block
    let mut out = BitVec::new();
    let data_iter = data.chunks(data_block_size);
    let mut parity_iter = parity.iter();
    for chunk in data_iter {
        out.extend(chunk.iter().chain(iter::once(
            parity_iter.next().expect("Insufficient parity bits"),
        )));
    }
    out
}

/// De-interleave parity bits into separate (data, parity).
#[must_use = "Doesn't affect inputs."]
pub fn deinterleave(interleaved: &BitSlice, data_block_size: usize) -> (BitVec, BitVec) {
    let mut parity = BitVec::new();
    let mut data = BitVec::new();

    // Block_size is number of data bits. +1 for parity bit.
    for block_with_parity in interleaved.chunks(data_block_size + 1) {
        // Use each block's individual length to handle remainder block at the end that isn't necessarily block_size.
        let (data_bits, parity_bit) = block_with_parity.split_at(block_with_parity.len() - 1);
        data.extend(data_bits);
        parity.extend(parity_bit);
    }

    (data, parity)
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
        let parity = generate_parity(&data, 5);
        assert_eq!(parity.len(), 3);
        assert_eq!(parity[0], true);
        assert_eq!(parity[1], false);
        assert_eq!(parity[2], false);

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
        let parity = generate_parity(&data, 5);
        assert_eq!(parity.len(), 2);
        assert_eq!(parity[0], true);
        assert_eq!(parity[1], false);
    }

    #[test]
    fn test_validate_parity() {
        let mut data = {
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
        let parity = generate_parity(&data, 5);
        let parity_errors = validate_parity(&data, &parity, 5);
        for (i, bit) in parity_errors.into_iter().enumerate() {
            assert_eq!(bit, false, "Failed on iteration {i}");
        }
        *data.get_mut(0).unwrap() = !data[0];
        let parity_errors = validate_parity(&data, &parity, 5);
        assert_eq!(parity_errors.len(), 3);
        assert_eq!(parity_errors[0], true);
        assert_eq!(parity_errors[1], false);

        let mut data = {
            let mut data = BitVec::new();
            for bit in [
                false, true, false, true, true, // 3
                true, true, false, false, false, // 2
            ] {
                data.push(bit);
            }
            data
        };
        let parity = generate_parity(&data, 5);
        let parity_errors = validate_parity(&data, &parity, 5);
        assert_eq!(parity_errors.len(), 2);
        for (i, bit) in parity_errors.into_iter().enumerate() {
            assert_eq!(bit, false, "Failed on iteration {i}");
        }
        *data.get_mut(0).unwrap() = !data[0];
        let parity_errors = validate_parity(&data, &parity, 5);
        assert_eq!(parity_errors[0], true);
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
        let parity = generate_parity(&data, 5);
        let interleaved = interleave(&data, &parity, 5);
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
        let parity = generate_parity(&data, 5);
        let interleaved = interleave(&data, &parity, 5);
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
        let parity = generate_parity(&data, 5);
        let interleaved = interleave(&data, &parity, 5);
        let deinterleaved = deinterleave(&interleaved, 5);
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
        let parity = generate_parity(&data, 5);
        let interleaved = interleave(&data, &parity, 5);
        let deinterleaved = deinterleave(&interleaved, 5);
        assert_eq!(deinterleaved.0, data);
        assert_eq!(deinterleaved.1, parity);
    }
}
