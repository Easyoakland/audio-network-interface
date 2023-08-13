// Constants that should eventually be substituted with cli args or runtime calculation.
pub const SHARD_BYTES_LEN: usize = 4;
pub const SHARD_BITS_LEN: usize = SHARD_BYTES_LEN * BITS_PER_BYTE;
pub const SENSITIVITY: f64 = 1.0;
pub const TIME_SAMPLES_PER_SYMBOL: usize = 4800;
/// Skip startup samples (number picked by experimentation at 40khz) because most device mics record silence when starting up
pub const SKIPPED_STARTUP_SAMPLES: usize = 20_000;

// Constants that make sense as constants.
pub const BITS_PER_BYTE: usize = u8::BITS as usize;
pub const REED_SOL_MAX_SHARDS: usize = 256;
