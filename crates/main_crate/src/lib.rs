/// Arguments for the various binaries.
/// This is compartmentalized to a module because the binaries all have almost the exact same argument requirements.
pub mod args;

/// Simple helper functions for reading and writing files.
pub mod file_io;

/// Plotting functionality.
#[cfg(feature = "plotters")]
pub mod plotting;

/// Functionality used primarily in transmission of data.
pub mod transmit;

/// Functionality used primarily in receiving and decoding data.
pub mod receive;
