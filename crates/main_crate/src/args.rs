use clap::{
    builder::{PossibleValuesParser, TypedValueParser as _},
    Parser,
};
use log::Level;

pub mod transmit_receive_args {
    use super::*;
    #[derive(Parser, Debug)]
    #[command(version, about = "Data transmission over audio.", long_about = None)]
    pub struct Args {
        /// The input file.
        #[arg(index = 1, required = true)]
        pub in_file: String,

        /// The output file.
        #[arg(short, long, default_value_t = String::from("out.out"))]
        pub out_file: String,

        /// The audio device to use.
        #[arg(short, long, default_value_t = String::from("default"))]
        pub device: String,

        /// The logging level to use.
        #[arg(
            short, long, default_value_t = Level::Info,
            // Needed because enum is foreign so can't use ValueEnum derive.
            value_parser = PossibleValuesParser::new(["trace", "debug", "info", "warn", "error"]).map(|s| s.parse::<Level>().unwrap()),
            ignore_case = true
        )]
        pub log_level: Level,

        /// The time per symbol in milliseconds.
        #[arg(short = 't', long, default_value_t = 100)]
        pub symbol_time: u64,

        /// The width allocated to a bit in hertz. Total bandwith for a byte is 8 * width.
        #[arg(short = 'w', long, default_value_t = 20.0)]
        pub bit_width: f32,

        /// The starting frequency of the transmission in hertz.
        #[arg(short, long, default_value_t = 1000.0)]
        pub start_freq: f32,

        /// The number of parallel channels to use for transmission.
        #[arg(short, long, default_value_t = 8)]
        pub parallel_channels: usize,
    }
}
