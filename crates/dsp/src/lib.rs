/// Encoder and decoder for on off keyed frequency division multiplexing.
pub mod ook_fdm;

/// Encoder and decoder for orthogonal frequency division multiplexing.
pub mod ofdm;

/// Encoder and decoder for on off keying.
pub mod carrier_modulation;

/// Correlation calculations. Used to determine how similar two signals are.
pub mod correlation;

/// Specs for different transmission types.
pub mod specs;

/// Conversion between bit (BitVec/Iterator<Item =bool>) and byte types (almost everything).
pub mod bit_byte_conversion;
