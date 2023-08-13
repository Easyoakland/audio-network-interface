use clap::{
    builder::{PossibleValuesParser, TypedValueParser as _},
    Args, Parser, Subcommand, ValueHint,
};
use dsp::specs::TransmissionSpec;
use log::Level;
use std::path::PathBuf;

/// Logging options.
#[derive(Args, Clone)]
pub struct LoggingOpt {
    /// The logging level to use.
    #[arg(
        short, long, default_value_t = Level::Info,
        // Needed because enum is foreign so can't use ValueEnum derive.
        value_parser = PossibleValuesParser::new(["trace", "debug", "info", "warn", "error"]).map(|s| s.parse::<Level>().unwrap()),
        ignore_case = true
    )]
    pub log_level: Level,
}

#[derive(Args, Clone)]
pub struct FileInOpt {
    /// The input file.
    #[arg(value_hint = ValueHint::FilePath)]
    pub in_file: PathBuf,
}

#[derive(Args, Clone)]
pub struct FileOutOpt {
    /// The output file.
    #[arg(value_hint = ValueHint::FilePath)]
    pub out_file: PathBuf,
}

/// Data transmission over audio.
#[derive(Parser, Clone)]
#[command(version)]
pub struct BaseCli {
    #[command(flatten)]
    pub log_opt: LoggingOpt,

    #[command(flatten)]
    pub file_in: FileInOpt,

    #[command(flatten)]
    pub file_out: FileOutOpt,
}

/// Forward error correction.
#[derive(Args, Clone)]
pub struct FecSpec {
    /// The number of parity shards to use as duplicate information.
    #[arg(short, long, default_value_t = 5)]
    pub parity_shards: usize,
}

/// Encoding and transmitting a signal.
#[derive(Args, Clone)]
pub struct TransmitOpt {
    #[command(flatten)]
    pub in_file: FileInOpt,

    /// The file to output to. If not provided outputs audio device.
    #[arg(value_hint = ValueHint::FilePath)]
    pub out_file: Option<PathBuf>,

    /// The forward error correction information
    #[command(flatten)]
    pub fec_spec: FecSpec,

    #[command(subcommand)]
    pub transmission_spec: TransmissionSpec,
}

/// Receiving and decoding a signal.
#[derive(Args, Clone)]
pub struct ReceiveOpt {
    #[command(flatten)]
    pub in_file: FileInOpt,

    #[command(flatten)]
    pub out_file: FileOutOpt,

    #[command(flatten)]
    pub fec_spec: FecSpec,

    #[command(subcommand)]
    pub transmission_spec: TransmissionSpec,
}

/// Transmit or receive options.
#[derive(Subcommand, Clone)]
pub enum TransceiverOpt {
    Transmit(TransmitOpt),
    Receive(ReceiveOpt),
}

#[derive(Parser, Clone)]
// default
#[doc = concat!(
    include_str!("about.md"),
    "\n\n",
    include_str!("../../../UsageInstructions.md")
)]
#[command(
    version,
    about = include_str!("about.md"),
    long_about = concat!(
        include_str!("about.md"),
        "\n\n",
        include_str!("../../../UsageInstructions.md")
    ),
)]
// wasm about
#[cfg_attr(
    target_arch = "wasm32",
    command(
        long_about = concat!(
            include_str!("about.md"),
            "\n",
            "UX and perf is better on native version due to wasm limitations.",
            "\n\n",
            include_str!("../../../UsageInstructions.md")
        ),
    ),
)]
pub struct TransmissionCli {
    #[command(flatten)]
    pub log_opt: LoggingOpt,

    #[command(subcommand)]
    pub transceiver_opt: TransceiverOpt,
    /*
       /// The audio device to use.
       #[arg(short, long, default_value_t = String::from("default"))]
       pub device: String,
    */
}
