// Constants that should eventually be substituted with cli args or runtime calculation.
pub const SHARD_BYTES_LEN: usize = 4;
pub const SHARD_BITS_LEN: usize = SHARD_BYTES_LEN * BITS_PER_BYTE;
pub const SENSITIVITY: f64 = 1.0;
pub const FIRST_BIN: usize = 20;
pub const SIMULTANEOUS_BYTES: usize = 8;
pub const TIME_SAMPLES_PER_SYMBOL: usize = 4800;

// Constants that make sense as constants.
pub const BITS_PER_BYTE: usize = 8;
pub const REED_SOL_MAX_SHARDS: usize = 256;
