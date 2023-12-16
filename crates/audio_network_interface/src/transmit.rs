use crate::{
    args::FecSpec,
    constants::{REED_SOL_MAX_SHARDS, SENSITIVITY, SHARD_BITS_LEN, SHARD_BYTES_LEN},
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
use log::{error, info, trace, warn};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    future::Future,
    io,
    iter::Iterator,
    marker::{Send, Sync},
    sync::{
        mpsc::{self, Receiver, RecvError, Sender, TryRecvError},
        Once,
    },
    task::{Poll, Waker},
    time::Duration,
};
use stft::{fft::window_fn, SpecCompute, WindowLength};
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
) -> Result<AsyncStream, StreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let is_done = mpsc::channel();
    let waker = mpsc::channel::<Waker>();
    let once = Once::new();

    let mut next_value = move || match signal.next() {
        Some(x) => x,
        None => {
            // Indicate end of signal.
            match is_done.0.send(()) {
                Ok(()) => {
                    once.call_once(|| trace!("Finished playing stream."));
                    if let Ok(waker) = waker.1.try_recv() {
                        waker.wake()
                    };
                }
                // Error only occurs if channel receiver end has closed.
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
    Ok(AsyncStream::new(stream, waker.0, is_done.1))
}

/// Future that returns [`Poll::Pending`] until the [`Stream`] runs out of samples.
#[must_use = "Stream will play until dropped"]
pub struct AsyncStream {
    /// Future holds the stream so it continues playing. When the future is dropped so is the stream.
    _stream: Stream,
    /// Send waker to stream through this channel. Stream decides when its done.
    waker_channel: Sender<Waker>,
    /// Stream lets this future know when it is done through this channel.
    is_done: Receiver<()>,
}

impl AsyncStream {
    pub fn new(stream: Stream, waker_channel: Sender<Waker>, is_done: Receiver<()>) -> Self {
        AsyncStream {
            _stream: stream,
            waker_channel,
            is_done,
        }
    }
}

impl Future for AsyncStream {
    type Output = Result<(), RecvError>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        // Try to send waker to the stream. Don't bother checking if successful. The stream will wake when finished.
        let _ = self.waker_channel.send(cx.waker().clone());
        match self.is_done.try_recv() {
            Ok(()) => Poll::Ready(Ok(())),
            Err(TryRecvError::Disconnected) => Poll::Ready(Err(RecvError)),
            Err(TryRecvError::Empty) => Poll::Pending,
        }
    }
}

/// Returns an [`AsyncStream`] that plays sound samples until dropped.
/// If the iterator runs out of samples it plays silence (equilibrium samples) and returns a [`Poll::Ready(())`].
pub fn play_stream(
    signal: impl Iterator<Item = f32> + Sync + Send + 'static,
) -> Result<AsyncStream, StreamError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(BuildStreamError::DeviceNotAvailable)?;
    let config = match device.default_output_config() {
        Ok(x) => Ok(x),
        Err(cpal::DefaultStreamConfigError::DeviceNotAvailable) => {
            Err(BuildStreamError::DeviceNotAvailable)
        }
        Err(cpal::DefaultStreamConfigError::StreamTypeNotSupported) => {
            Err(BuildStreamError::StreamConfigNotSupported)
        }
        Err(cpal::DefaultStreamConfigError::BackendSpecific { err }) => {
            Err(BuildStreamError::BackendSpecific { err })
        }
    }?;

    trace!(
        "Output device: {}",
        device
            .name()
            .unwrap_or_else(|e| format!("no device name. {e}"))
    );
    trace!("Default output config: {:?}", config);

    // Select the correct data type depending on supported format and then runs.
    match config.sample_format() {
        SampleFormat::I8 => run::<i8>(&device, &config.into(), signal),
        SampleFormat::I16 => run::<i16>(&device, &config.into(), signal),
        SampleFormat::I32 => run::<i32>(&device, &config.into(), signal),
        SampleFormat::I64 => run::<i64>(&device, &config.into(), signal),
        SampleFormat::U8 => run::<u8>(&device, &config.into(), signal),
        SampleFormat::U16 => run::<u16>(&device, &config.into(), signal),
        SampleFormat::U32 => run::<u32>(&device, &config.into(), signal),
        SampleFormat::U64 => run::<u64>(&device, &config.into(), signal),
        SampleFormat::F32 => run::<f32>(&device, &config.into(), signal),
        SampleFormat::F64 => run::<f64>(&device, &config.into(), signal),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
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
pub async fn encode_transmission<O, E, F>(
    fec_spec: FecSpec,
    transmission_spec: TransmissionSpec,
    bytes: impl DynCloneIterator<u8> + Clone + 'static,
    mut sink: impl FnMut(Box<dyn DynCloneIterator<f32>>) -> Result<F, E>,
) -> Result<O, EncodingError<E>>
where
    E: Sync + Send + std::error::Error + 'static,
    F: Future<Output = O>,
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

        info!("Output device sample rate: {sample_rate}Hz. Record at this sample rate (preferred) or higher.");

        // Encode appropriately.
        let encoded_signal: Box<dyn DynCloneIterator<f32>> = match transmission_spec {
            TransmissionSpec::Fdm(fdm) => {
                Box::new(transmit::encode_fdm(fdm, sample_rate as f32, bits))
            }
            TransmissionSpec::Ofdm(ofdm_spec) => {
                // Create subcarrier encoding layout.
                // TODO don't need leak. Need to optionally create value in outer scope.
                let subcarriers_encoders =
                    vec![SubcarrierEncoder::T0(null_encode); ofdm_spec.bin_num()].leak();
                for i in ofdm_spec.active_bins() {
                    subcarriers_encoders[i] = SubcarrierEncoder::T1(bpsk_encode);
                }
                // Use the encoding layout to actually create the frames with samples.
                let frames = OfdmFramesEncoder::new(bits, subcarriers_encoders, ofdm_spec);
                trace!("Encoded {} frames.", frames.clone().count());
                Box::new(frames.flatten())
            }
        };

        // Play `encoded_signal`.
        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();
        let res = sink(encoded_signal)?.await;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let end = Instant::now();
            trace!("Played stream for {:?}", (end - start));
        }

        Ok(res)
    }
}

#[derive(Error, Debug)]
pub enum DecodingError {
    #[error("No frame found: {0}")]
    NoFrame(String),
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
    sample_rate: f32,
) -> Result<Vec<u8>, DecodingError> {
    // Skip startup samples
    let source = source.skip_while(|&x| x == 0.);

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
            let mut subcarriers_decoders =
                vec![SubcarrierDecoder::Data(null_decode); ofdm_spec.bin_num()];
            for i in ofdm_spec.active_bins() {
                subcarriers_decoders[i] = SubcarrierDecoder::Data(bpsk_decode);
            }
            // Generate transmitted preamble for comparison.
            let tx_preamble = ofdm_preamble_encode(&ofdm_spec).collect::<Vec<_>>();
            // Detect start of frame by comparing to reference preamble.
            // TODO don't collect all source into a vec. Either reimplement detector or use a lazy collection.
            let Some(frame_start) = ofdm_premable_cross_correlation_detector(
                &source.clone().collect::<Vec<_>>(),
                &tx_preamble[..ofdm_spec.time_symbol_len() / ofdm_spec.short_training_repetitions],
                ofdm_spec.cross_correlation_threshold,
            ) else {
                return Err(DecodingError::NoFrame(
                    "Can't find start of frame.".to_owned(),
                ));
            };
            trace!("Frame Start: {:?}", frame_start);
            // Setup decoder.
            let frames_decoder = OfdmFramesDecoder::new(
                source.into_iter().skip(frame_start.0),
                &subcarriers_decoders,
                ofdm_spec,
            );
            // Flatten all frames of bits into a really long bitvector
            // TODO this collects the entire transmission into memory
            frames_decoder.flatten().collect::<BitVec<u8, Lsb0>>()
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
