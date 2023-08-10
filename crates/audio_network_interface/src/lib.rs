/// Arguments for the various binaries.
pub mod args;

/// Simple helper functions for reading and writing files.
pub mod file_io;

/// Plotting functionality.
#[cfg(feature = "plot")]
pub mod plotting;

/// Functionality used primarily in transmission of data.
pub mod transmit;

/// Constants used for various purposes.
pub mod constants;

/// The main functions for the binaries. Useful for reusing in gui/nongui/etc...
pub mod binary_logic;

/// Tests.
#[cfg(test)]
pub mod tests;
