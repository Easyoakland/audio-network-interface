use crate::traits::Function;
use itertools::Itertools;
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::{iter, sync::Arc};

/// Encodes data with reed solomon erasure encoding
#[derive(Debug, Default, Clone)]
pub struct ReedSolomonEncoder {
    /// Size of blocks in bytes.
    pub block_size: usize,
    /// Number of parity blocks.
    pub parity_blocks: usize,
}

/// Reuse the created [`ReedSolomon`] since creation is a high compute operation.
/// Would clone directly, but clone is implemented in terms of [`ReedSolomon::new`] and therefore has the same high compute cost.
#[cached::proc_macro::cached]
fn cached_new_reed_sol(data_blocks: usize, parity_blocks: usize) -> Arc<ReedSolomon> {
    Arc::new(ReedSolomon::new(data_blocks, parity_blocks).unwrap())
}

impl ReedSolomonEncoder {
    /// Encodes the data into shards with the given number of parity blocks.
    pub fn encode(&self, data: Vec<u8>) -> Result<Vec<Vec<u8>>, reed_solomon_erasure::Error> {
        // Count the qty of block sizes that fully fit the data +1 for the padded shard.
        let data_blocks = data.len() / self.block_size + 1;
        // Will add an extra shard/block of the form `[padding_zeros_qty, 0, 0, ...]`.
        // `padding` is that first byte. It is the amount of extra zeros after itself.
        let padding = self.block_size - data.len() % self.block_size - 1;

        // Define encoder.
        let r = cached_new_reed_sol(data_blocks, self.parity_blocks);

        // Add padding amount and padding
        let data = iter::once(padding as u8)
            .chain(iter::repeat(0).take(padding))
            .chain(data);

        // Add each block of data as a shard.
        let mut shards: Vec<Vec<u8>> = Vec::new();
        for block in &data.chunks(self.block_size) {
            shards.push(block.collect());
        }

        // Add parity placeholders
        for _ in 0..self.parity_blocks {
            shards.push(vec![Default::default(); self.block_size]);
        }

        // Set parity bits.
        r.encode(&mut shards)?;
        Ok(shards)
    }
}

impl Function for ReedSolomonEncoder {
    type Input = Vec<u8>;

    type Output = Result<Vec<Vec<u8>>, reed_solomon_erasure::Error>;

    fn map(&self, input: Self::Input) -> Self::Output {
        self.encode(input)
    }
}

/// Decodes and reconstructs data with reed solomon erasure encoding
#[derive(Debug, Default, Clone)]
pub struct ReedSolomonDecoder {
    pub parity_shards: usize,
}

impl ReedSolomonDecoder {
    /// Reconstructs data from partially erased shards.
    pub fn reconstruct_data(
        &self,
        mut data: Vec<Option<Vec<u8>>>,
    ) -> Result<Vec<u8>, reed_solomon_erasure::Error> {
        let data_shards = data
            .len()
            .checked_sub(self.parity_shards)
            .ok_or(reed_solomon_erasure::Error::TooFewShards)?;

        let r = cached_new_reed_sol(data_shards, self.parity_shards);

        // If padding shard exists make sure it has a valid amount of padding.
        // Remove padding. Remember to skip the amount byte indicating the amount of padding in addition to the extra 0's.
        // If the shard appears incorrect replace it with None before reconstruction.
        if let Some(padded_shard) = data[0].clone() {
            let padding = padded_shard[0] as usize;
            if padding > padded_shard.len() {
                data[0] = None;
                log::warn!(
                    "Should not have more padding ({padding}) than the length of a shard ({})",
                    padded_shard.len()
                );
            }
        }

        r.reconstruct_data(&mut data)?; // No None variant in data shards after here.

        // Serialize the `Vec<Option::Some<Vec<u8>>>` into `Vec<u8>`
        let out = data
            .into_iter()
            .flatten()
            .take(data_shards)
            .flatten()
            .collect::<Vec<_>>();
        let padding = out[0];

        Ok(out
            .into_iter()
            .skip((1 + padding).into()) // Skip padding amount indicator and the padding zeros
            .collect())
    }
}

impl Function for ReedSolomonDecoder {
    type Input = Vec<Option<Vec<u8>>>;

    type Output = Result<Vec<u8>, reed_solomon_erasure::Error>;

    fn map(&self, input: Self::Input) -> Self::Output {
        self.reconstruct_data(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::reed_solomon::{ReedSolomonDecoder, ReedSolomonEncoder};

    #[test]
    fn test_reed_solomon_padding_unneeded() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let encoder = ReedSolomonEncoder {
            block_size: 4,
            parity_blocks: 4,
        };
        let decoder = ReedSolomonDecoder { parity_shards: 4 };
        let encoded = encoder.encode(data.clone()).unwrap();
        let encoded: Vec<_> = encoded.into_iter().map(Some).collect();
        let decoded = decoder.reconstruct_data(encoded.clone());
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
        let decoded_recovered = decoder.reconstruct_data(encoded_damaged);
        assert_eq!(decoded, decoded_recovered);
    }

    #[test]
    fn test_reed_solomon_padding_needed() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let encoder = ReedSolomonEncoder {
            block_size: 4,
            parity_blocks: 4,
        };
        let decoder = ReedSolomonDecoder { parity_shards: 4 };
        let encoded = encoder.encode(data.clone()).unwrap();
        let encoded: Vec<_> = encoded.into_iter().map(Some).collect();
        let decoded = decoder.reconstruct_data(encoded.clone());
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
        let decoded_recovered = decoder.reconstruct_data(encoded_damaged);
        assert_eq!(decoded, decoded_recovered);
    }
}
