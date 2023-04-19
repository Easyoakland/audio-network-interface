use clap::{
    builder::{PossibleValuesParser, TypedValueParser as _},
    Args, Parser, Subcommand,
};
use dsp::specs::{FdmSpec, OfdmSpec};
use log::Level;

#[derive(Parser, Clone)]
#[command(version)]
/// Data transmission over audio.
pub struct BaseCli {
    #[command(flatten)]
    pub log_opt: LoggingOpt,

    #[command(flatten)]
    pub file_opt: FileOpt,
}

#[derive(Parser, Clone)]
#[command(version)]
/// Data transmission over audio.
pub struct TransmissionCli {
    #[command(flatten)]
    pub base: BaseCli,

    /*     /// The audio device to use.
       #[arg(short, long, default_value_t = String::from("default"))]
       pub device: String,
    */
    /// The forward error correction information
    #[command(flatten)]
    pub fec_spec: FecSpec,

    /// The spec for a transmission.
    // TODO Include things like realtime_vs_file
    #[command(subcommand)]
    pub transmission_spec: TransmissionSpec,
}

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
pub struct FileOpt {
    /// The input file.
    #[arg(index = 1, required = true)]
    pub in_file: String,

    /// The output file.
    #[arg(short, long)]
    pub out_file: String,
}

#[derive(Args, Clone)]
pub struct FecSpec {
    #[arg(short, long, default_value_t = 5)]
    pub parity_shards: usize,
}

#[derive(Subcommand, Clone)]
pub enum TransmissionSpec {
    Fdm(FdmSpec),
    Ofdm(OfdmSpec),
}
