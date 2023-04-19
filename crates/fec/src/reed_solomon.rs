use crate::traits::Function;
use itertools::Itertools;
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::iter;

/// Encodes data with reed solomon erasure encoding
#[derive(Debug, Default, Clone)]
pub struct ReedSolomonEncoder {
    pub block_size: usize,
    pub parity_blocks: usize,
}

impl ReedSolomonEncoder {
    /// Encodes the data into shards and with the given number of parity blocks.
    /// Returns the padding amount followed by padding followed by the encoding.
    pub fn encode(&self, data: Vec<u8>) -> Result<Vec<Vec<u8>>, reed_solomon_erasure::Error> {
        let padding;
        // Count data_blocks.
        let data_blocks = {
            let mut out = 0;
            // Add the qty of block sizes that fully fit the data +1 for the padded shard.
            out += data.len() / self.block_size + 1;
            // Add an extra block. Padding is amount of extra zeros after padding number.
            padding = self.block_size - data.len() % self.block_size - 1;
            out
        };

        // Define encoder.
        let r = ReedSolomon::new(data_blocks, self.parity_blocks).unwrap();

        // Add padding amount and padding
        let data = iter::once(padding as u8)
            .chain(iter::repeat(0).take(padding))
            .chain(data);

        // Add each block of data as a shard.
        let mut shards: Vec<Vec<u8>> = Vec::new();
        for block in data.chunks(self.block_size).into_iter() {
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
        const PADDED_DATA_SHARDS: usize = 1;
        let non_paddded_data_shards = data
            .len()
            .checked_sub(PADDED_DATA_SHARDS)
            .ok_or(reed_solomon_erasure::Error::TooFewShards)?
            .checked_sub(self.parity_shards)
            .ok_or(reed_solomon_erasure::Error::TooFewShards)?;

        // Reconstruct from encoding.
        let r = ReedSolomon::new(
            non_paddded_data_shards + PADDED_DATA_SHARDS,
            self.parity_shards,
        )
        .unwrap();

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
            .take(non_paddded_data_shards + PADDED_DATA_SHARDS)
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
