use itertools::Itertools;
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::iter;

/// Encodes the data into shards and with the given number of parity blocks.
/// Returns the padding amount followed by padding followed by the encoding.
pub fn encode(
    data: Vec<u8>,
    block_size: usize,
    parity_blocks: usize,
) -> Result<Vec<Vec<u8>>, reed_solomon_erasure::Error> {
    let padding;
    // Count data_blocks.
    let data_blocks = {
        let mut out = 0;
        // Add the qty of block sizes that fully fit the data +1 for the padded shard.
        out += data.len() / block_size + 1;
        // Add an extra block. Padding is amount of extra zeros after padding number.
        padding = block_size - data.len() % block_size - 1;
        out
    };

    // Define encoder.
    let r = ReedSolomon::new(data_blocks, parity_blocks).unwrap();

    // Add padding amount and padding
    let data = iter::once(padding as u8)
        .chain(iter::repeat(0).take(padding))
        .chain(data);

    // Add each block of data as a shard.
    let mut shards: Vec<Vec<u8>> = Vec::new();
    for block in data.chunks(block_size).into_iter() {
        shards.push(block.collect());
    }

    // Add parity placeholders
    for _ in 0..parity_blocks {
        shards.push(vec![Default::default(); block_size]);
    }

    // Set parity bits.
    r.encode(&mut shards)?;
    Ok(shards)
}

/// Reconstructs data from partially erased shards.
pub fn reconstruct_data(
    mut data: Vec<Option<Vec<u8>>>,
    parity_blocks: usize,
) -> Result<Vec<u8>, reed_solomon_erasure::Error> {
    let data_blocks = data.len() - parity_blocks;

    // Reconstruct from encoding.
    let r = ReedSolomon::new(data_blocks, parity_blocks).unwrap();
    r.reconstruct_data(&mut data)?; // No None variant after here.

    let padding_upper_bound = (*data.iter().find(|&x| x.is_some()).unwrap())
        .as_ref()
        .unwrap()
        .len();

    // Serialize the `Vec<Option::Some<Vec<u8>>>` into `Vec<u8>`
    let out = data
        .into_iter()
        .flatten()
        .take(data_blocks)
        .flatten()
        .collect::<Vec<_>>();

    // Remove padding. Remember to skip the amount of padding in addition to the extra 0's.
    let padding = out[0] as usize;
    assert!(
            padding <= padding_upper_bound,
            "Should not have more padding ({padding}) than the length of a shard ({padding_upper_bound})"
        );
    Ok(out.into_iter().skip(padding + 1).collect())
}

#[cfg(test)]
mod tests {
    use super::{encode, reconstruct_data};

    #[test]
    fn test_reed_solomon_padding_unneeded() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let encoded = encode(data.clone(), 4, 4).unwrap();
        let encoded: Vec<_> = encoded.into_iter().map(Some).collect();
        let decoded = reconstruct_data(encoded.clone(), 4);
        assert_eq!(data, decoded.clone().unwrap());
        // Should be able to do parity bits worth of damage and recover.
        let encoded_damaged = {
            let mut out = encoded;
            out[4] = None;
            out[1] = None;
            out[2] = None;
            out[3] = None;
            out
        };
        let decoded_recovered = reconstruct_data(encoded_damaged, 4);
        assert_eq!(decoded, decoded_recovered);
    }

    #[test]
    fn test_reed_solomon_padding_needed() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let encoded = encode(data.clone(), 4, 4).unwrap();
        let encoded: Vec<_> = encoded.into_iter().map(Some).collect();
        let decoded = reconstruct_data(encoded.clone(), 4);
        assert_eq!(data, decoded.clone().unwrap());
        // Should be able to do parity bits worth of damage and recover.
        let encoded_damaged = {
            let mut out = encoded;
            out[4] = None;
            out[1] = None;
            out[2] = None;
            out[3] = None;
            out
        };
        let decoded_recovered = reconstruct_data(encoded_damaged, 4);
        assert_eq!(decoded, decoded_recovered);
    }
}
