use crate::{
    args::FecSpec,
    constants::{
        REED_SOL_MAX_SHARDS, SENSITIVITY, SHARD_BITS_LEN, SHARD_BYTES_LEN, SIMULTANEOUS_BYTES,
        TIME_SAMPLES_PER_SYMBOL,
    },
    transmit,
};
use bitvec::{prelude::Lsb0, vec::BitVec};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, Device, FromSample, PlayStreamError, Sample, SampleFormat, SizedSample,
    Stream, StreamConfig,
};
use dsp::{
    bit_byte_conversion::bytes_to_bits,
    carrier_modulation::{bpsk_decode, bpsk_encode, null_decode, null_encode},
    ofdm::{
        ofdm_preamble_encode, ofdm_premable_cross_correlation_detector, OfdmFramesDecoder,
        OfdmFramesEncoder, SubcarrierDecoder, SubcarrierEncoder,
    },
    ook_fdm::{OokFdmConfig, OokFdmDecoder},
    specs::{FdmSpec, TransmissionSpec},
};
use dyn_clone::{clone_trait_object, DynClone};
use fec::{
    parity::{ParityDecoder, ParityEncoder},
    reed_solomon::{ReedSolomonDecoder, ReedSolomonEncoder},
    traits::Function,
};
use iterator_adapters::IteratorAdapter;
use log::{error, trace, warn};
use std::{
    io,
    iter::Iterator,
    marker::{Send, Sync},
    sync::mpsc::{self, RecvError, SyncSender, TrySendError},
    time::{Duration, Instant},
};
use stft::{fft::window_fn, time_samples_to_frequency, SpecCompute, WindowLength};
use thiserror::Error;

/// Writes the data given by the closure onto the output stream.
fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

/// Errors that can happen while building or playing a stream.
#[derive(Error, Debug)]
pub enum StreamError {
    #[error("Play stream error: {0}")]
    PlayStreamError(#[from] PlayStreamError),
    #[error("Build stream error: {0}")]
    BuildStreamError(#[from] BuildStreamError),
}

/// Main logic for sending sounds through the speaker.
fn run<T>(
    device: &Device,
    config: &StreamConfig,
    mut signal: impl Iterator<Item = f32> + Send + Sync + 'static,
    finished_indicator_channel: SyncSender<()>,
) -> Result<MustUse<Stream>, StreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;

    let mut next_value = move || match signal.next() {
        Some(x) => x,
        None => {
            // Indicate end of signal if channel to notify exists.
            match finished_indicator_channel.try_send(()) {
                Ok(()) => trace!("Finished playing stream."),
                // Full error only occurs if already sent signal.
                Err(TrySendError::Full(())) => (),
                Err(e) => error!("Error submitting end of signal: {e}"),
            }
            Sample::EQUILIBRIUM
        }
    };

    let err_fn = |err| error!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value);
        },
        err_fn,
        None,
    )?;
    trace!("Playing stream");
    stream.play()?;
    Ok(MustUse(stream))
}

/// Returns a [`Stream`] that plays sound samples until the returned stream is dropped.
/// If the iterator runs out it plays silence (equilibrium samples) and sends `()` through the channel.
pub fn play_stream(
    signal: impl Iterator<Item = f32> + Sync + Send + 'static,
    tx: SyncSender<()>,
) -> Result<MustUse<Stream>, StreamError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let config = device
        .default_output_config()
        .expect("no default config for device");

    trace!(
        "Output device: {}",
        device
            .name()
            .unwrap_or_else(|e| format!("no device name. {e}"))
    );
    trace!("Default output config: {:?}", config);

    // Select the correct data type depending on supported format and then runs.
    match config.sample_format() {
        SampleFormat::I8 => run::<i8>(&device, &config.into(), signal, tx),
        SampleFormat::I16 => run::<i16>(&device, &config.into(), signal, tx),
        SampleFormat::I32 => run::<i32>(&device, &config.into(), signal, tx),
        SampleFormat::I64 => run::<i64>(&device, &config.into(), signal, tx),
        SampleFormat::U8 => run::<u8>(&device, &config.into(), signal, tx),
        SampleFormat::U16 => run::<u16>(&device, &config.into(), signal, tx),
        SampleFormat::U32 => run::<u32>(&device, &config.into(), signal, tx),
        SampleFormat::U64 => run::<u64>(&device, &config.into(), signal, tx),
        SampleFormat::F32 => run::<f32>(&device, &config.into(), signal, tx),
        SampleFormat::F64 => run::<f64>(&device, &config.into(), signal, tx),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
}

/// Annotates type as must use.
/// Useful when the inner type (eg. of a result) must be used even if the outer type is used (eg. result contains important data that must bind).
#[must_use = "The inner type must be used."]
pub struct MustUse<T>(T);

impl<T> From<T> for MustUse<T> {
    fn from(v: T) -> Self {
        Self(v)
    }
}

/// Errors that can happen while blocking on building or playing a stream.
#[derive(Error, Debug)]
pub enum BlockingStreamError {
    #[error("Stream Error: {0}")]
    StreamError(#[from] StreamError),
    #[error("Stream send half disconnected before notifying finished playing")]
    RecvError,
}

impl From<RecvError> for BlockingStreamError {
    fn from(_: RecvError) -> Self {
        BlockingStreamError::RecvError
    }
}

/// Plays a signal and blocks until it is finished playing.
pub fn play_stream_blocking(
    signal: impl Iterator<Item = f32> + Send + Sync + 'static,
) -> Result<(), BlockingStreamError> {
    let (tx, rx) = mpsc::sync_channel(0);
    let _stream = play_stream(signal, tx)?;
    // Block while waiting for end signal.
    rx.recv().map_err(Into::into)
}

/// Generates an FDM ook transmission.
pub fn encode_fdm(
    fdm_spec: FdmSpec,
    sample_rate: f32,
    bits: impl Iterator<Item = bool> + Clone + Send + Sync + 'static,
) -> impl Iterator<Item = f32> + Clone {
    // Generate speaker amplitude iterator.
    // TODO replace leaking with scoped thread.
    // Not a big deal because config only needs to be made once to use for all future signals.
    let config = Box::leak(Box::new(OokFdmConfig::new(
        fdm_spec.bit_width,
        fdm_spec.parallel_channels,
        sample_rate,
        fdm_spec.start_freq,
        WindowLength::from_duration(Duration::from_millis(fdm_spec.symbol_time), sample_rate)
            .samples(),
    )));
    config.encode(bits)
}

/// Iterator that can be used as a trait object **and** cloned.
pub trait DynCloneIterator<T>: Iterator<Item = T> + Send + Sync + DynClone {}
clone_trait_object!(<T> DynCloneIterator<T>);
impl<T, I: Iterator<Item = T> + Send + Sync + Clone> DynCloneIterator<T> for I {}

const PADDED_SHARDS: usize = 1;
/// Represents number of bytes that can be used in one chunk of reed sol encoding.
const fn reed_sol_chunk_byte_size(parity_shards: usize) -> usize {
    (REED_SOL_MAX_SHARDS * SHARD_BYTES_LEN)
        - parity_shards * SHARD_BYTES_LEN
        - PADDED_SHARDS * SHARD_BYTES_LEN
}

#[derive(Error, Debug)]
pub enum EncodingError<E> {
    #[error("Forward error correction failed to encode: {0}")]
    Fec(String),
    #[error(transparent)]
    Default(#[from] E),
}

/// Logic for sending a data transmission to a data sink.
pub fn encode_transmission<E>(
    fec_spec: FecSpec,
    transmission_spec: TransmissionSpec,
    bytes: impl DynCloneIterator<u8> + Clone + 'static,
    mut sink: impl FnMut(Box<dyn DynCloneIterator<f32>>) -> Result<(), E>,
) -> Result<(), EncodingError<E>>
where
    E: Sync + Send + std::error::Error + 'static,
{
    {
        let reed_encoder = ReedSolomonEncoder {
            block_size: SHARD_BYTES_LEN,
            parity_blocks: fec_spec.parity_shards,
        };
        let parity_encoder = ParityEncoder {
            data_block_size: SHARD_BITS_LEN,
        };

        // Encode with reed_solomon.
        let reed_encoding = bytes
            .chunks(reed_sol_chunk_byte_size(fec_spec.parity_shards))
            .map(move |x| reed_encoder.map(x));
        // Add parity checks to shards and convert bytes to bits.
        let bits = reed_encoding
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| EncodingError::Fec(format!("Reed solomon encoding failed: {e}")))?
            .into_iter()
            .flat_map(move |x| parity_encoder.map(x.into_iter().flatten().collect()));
        trace!("Transmitting {} bits.", bits.clone().count());

        // Get speaker sample_rate.
        let sample_rate = cpal::default_host()
            .default_output_device()
            .expect("no output device available")
            .default_output_config()
            .unwrap()
            .sample_rate()
            .0;

        // Encode appropriately.
        let encoded_signal: Box<dyn DynCloneIterator<f32>> = match transmission_spec {
            TransmissionSpec::Fdm(fdm) => {
                Box::new(transmit::encode_fdm(fdm, sample_rate as f32, bits))
            }
            TransmissionSpec::Ofdm(ofdm_spec) => {
                // Create subcarrier encoding layout.
                let subcarriers_encoders = Box::leak(Box::new(
                    [SubcarrierEncoder::T0(null_encode);
                        time_samples_to_frequency(TIME_SAMPLES_PER_SYMBOL)],
                ));
                let active_bins =
                    ofdm_spec.first_bin..(ofdm_spec.first_bin + 8 * SIMULTANEOUS_BYTES);
                for i in active_bins {
                    subcarriers_encoders[i] = SubcarrierEncoder::T1(bpsk_encode);
                }
                // Use the encoding layout to actually create the frames with samples.
                let frames = OfdmFramesEncoder::new(bits, subcarriers_encoders, ofdm_spec);
                trace!("Encoded {} frames.", frames.clone().count());
                Box::new(frames.flatten())
            }
        };

        // Play `encoded_signal`.
        let start = Instant::now();
        sink(encoded_signal)?;
        let end = Instant::now();
        trace!("Played stream for {:?}", (end - start));

        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum DecodingError {
    #[error("No packet found: {0}")]
    NoPacket(String),
    #[error("Io Error: {0}")]
    Io(#[from] io::Error),
    #[error("Forward error correction failed to decode: {0}")]
    Fec(String),
}

/// Logic for decoding the data from a channel source.
pub fn decode_transmission(
    fec_spec: FecSpec,
    transmission_spec: TransmissionSpec,
    source: impl DynCloneIterator<f32> + Clone,
    startup_samples_to_skip: usize,
    sample_rate: f32,
) -> Result<Vec<u8>, DecodingError> {
    // Skip startup samples
    let source = source.skip(startup_samples_to_skip);

    // Get data using correct method.
    let bits: BitVec<u8, Lsb0> = match transmission_spec {
        TransmissionSpec::Fdm(fdm_spec) => {
            // Compute stft of data.
            let window_len = WindowLength::from_duration(
                Duration::from_millis(fdm_spec.symbol_time),
                sample_rate,
            );
            let spec_compute = SpecCompute::new(
                source.map(f64::from).collect(),
                window_len,
                window_len,
                window_fn::hann,
            );
            let stft = spec_compute.stft();

            // Setup transmission parameters.
            let frequency_channels = (0..fdm_spec.parallel_channels)
                .map(|i| fdm_spec.start_freq + (1 + i) as f32 * fdm_spec.bit_width)
                .collect();
            let decoder = OokFdmDecoder {
                frequency_channels,
                sample_rate,
            };

            // Decode the transmission.
            bytes_to_bits(decoder.decode_ook_fdm(&stft, SENSITIVITY).into_iter()).collect()
        }
        TransmissionSpec::Ofdm(ofdm_spec) => {
            // Describe subcarrier encoding.
            let subcarriers_decoders = Box::leak(Box::new(
                [SubcarrierDecoder::Data(null_decode);
                    time_samples_to_frequency(TIME_SAMPLES_PER_SYMBOL)],
            ));
            let active_bins = ofdm_spec.first_bin..(ofdm_spec.first_bin + 8 * SIMULTANEOUS_BYTES);
            for i in active_bins {
                subcarriers_decoders[i] = SubcarrierDecoder::Data(bpsk_decode);
            }
            // Generate transmitted preamble for comparison.
            let tx_preamble = ofdm_preamble_encode(&ofdm_spec).collect::<Vec<_>>();
            // Detect start of frame by comparing to reference preamble.
            let Some(packet_start) = ofdm_premable_cross_correlation_detector(
                &source.clone().collect::<Vec<_>>(),
                &tx_preamble
                    [..ofdm_spec.time_symbol_len / ofdm_spec.short_training_repetitions],
                ofdm_spec.cross_correlation_threshold,
            ) else {
                return Err(DecodingError::NoPacket("Can't find start of packet.".to_owned()));
            };
            trace!("Packet Start: {:?}", packet_start);
            // Setup decoder.
            let frames_decoder = OfdmFramesDecoder::new(
                source.into_iter().skip(packet_start.0),
                *subcarriers_decoders,
                ofdm_spec,
            );
            frames_decoder
                .flat_map(|x| x.bits_to_bytes::<u8>())
                .collect()
        }
    };

    let parity_decoder = ParityDecoder {
        data_block_size: SHARD_BITS_LEN,
    };
    let reed_decoder = ReedSolomonDecoder {
        parity_shards: fec_spec.parity_shards,
    };

    // Decode fec.
    let shards = parity_decoder.map(bits);
    let bytes = shards
        .into_iter()
        .chunks(REED_SOL_MAX_SHARDS) // decode each chunk
        .enumerate()
        .flat_map(|(i, x)| {
            let temp = reed_decoder.map(x.clone()); // with reed solomon
            match temp {
                Ok(x) => x,
                Err(e) => {
                    warn!("Reed solomon decoding failed on block {i}: {e}");
                    x.into_iter()
                        .flat_map(|shard| match shard {
                            Some(x) => x,                     // using correct values
                            None => vec![0; SHARD_BYTES_LEN], // and replacing values that couldn't be corrected with null (0 byte) for the number of missing values
                        })
                        .collect()
                }
            }
        })
        .collect();
    Ok(bytes)
}
