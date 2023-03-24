use bitvec::{field::BitField, vec::BitVec};
use nalgebra::SMatrix;

/// Encodes data with hamming74
pub fn hamming74_encode(data: BitVec) -> BitVec {
    #[rustfmt::skip]
        let gt = SMatrix::<u8, 7, 4>::from_row_slice(&[
            1,1,0,1,
            1,0,1,1,
            1,0,0,0,
            0,1,1,1,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1,
        ]);
    let data = data.iter().by_vals().map(|b| if b { 1 } else { 0 });
    let data_block = SMatrix::<_, 4, 1>::from_iterator(data.take(4));
    let codeword = {
        let mut codeword = gt * data_block;
        codeword.iter_mut().for_each(|x| *x %= 2);
        codeword
    };
    codeword.iter().map(|x| x % 2 != 0).collect()
}

/// Calculates parity check data with hamming74
fn hamming74_parity_check(data: BitVec) -> BitVec {
    #[rustfmt::skip]
        let h= SMatrix::<u8, 3, 7>::from_row_slice(&[
            1,0,1,0,1,0,1,
            0,1,1,0,0,1,1,
            0,0,0,1,1,1,1,
        ]);
    let data = data.iter().by_vals().map(|b| if b { 1 } else { 0 });
    let data_block = SMatrix::<u8, 7, 1>::from_iterator(data.take(8));
    let codeword = h * data_block;
    codeword.iter().map(|x| x % 2 != 0).collect()
}

/// Returns hamming74 encoded message with all detected errors fixed.
fn hamming74_fix(mut data: BitVec) -> BitVec {
    let error = hamming74_parity_check(data.clone());
    let err_idx = match usize::from(error.load_le::<u8>()) {
        0 => return data, // 0 indicates no error
        x => x - 1,       // nonzero indicates error starting from 1
    };
    *data.get_mut(err_idx).unwrap() = !data.get(err_idx).unwrap();
    data
}

/// Decodes a hamming74 encoded message.
pub fn hamming74_decode(data: BitVec) -> BitVec {
    let data = hamming74_fix(data);
    #[rustfmt::skip]
        let r = SMatrix::<_, 4, 7>::from_row_slice(&[
            0,0,1,0,0,0,0,
            0,0,0,0,1,0,0,
            0,0,0,0,0,1,0,
            0,0,0,0,0,0,1,
        ]);
    let data = data.iter().by_vals().map(|b| if b { 1 } else { 0 });
    let data_block = SMatrix::<u8, 7, 1>::from_iterator(data.take(8));
    let codeword = r * data_block;
    codeword.iter().map(|x| x % 2 != 0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::{bitvec, prelude::Lsb0};
    use proptest::proptest;
    proptest! {
        #[test]
        fn hamming74_test(err_idx in 0..7usize) {
            let data = bitvec![1, 0, 1, 1];
            let hamm_encoded = hamming74_encode(data.clone());
            assert_eq!(hamm_encoded, bitvec![0, 1, 1, 0, 0, 1, 1]);
            assert_eq!(
                hamming74_parity_check(hamm_encoded.clone()),
                bitvec![0, 0, 0]
            );
            let hamm_encoded_err = {
                let mut invalid = hamm_encoded.clone();
                *invalid.get_mut(err_idx).unwrap() = !invalid.get(err_idx).unwrap();
                invalid
            };
            assert_eq!(hamming74_fix(hamm_encoded_err.clone()), hamm_encoded);
            assert_eq!(
                hamming74_fix(hamm_encoded_err.clone()),
                hamming74_fix(hamm_encoded.clone())
            );
            assert_eq!(hamming74_decode(hamm_encoded_err.clone()), data);
            assert_eq!(
                hamming74_decode(hamm_encoded_err),
                hamming74_decode(hamm_encoded)
            );
        }
    }
}
