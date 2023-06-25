#![cfg(feature = "plot")]
#![allow(unused_imports, unused_variables)]
use audio_network_interface::{
    file_io::read_wav,
    plotting::{plot_series, scatterplot},
    transmit::bytes_to_bits,
};
use bitvec::{prelude::Lsb0, vec::BitVec};
use rand_distr::{Distribution, Normal};
// use cpal::traits::{DeviceTrait as _, HostTrait as _};
use dsp::{
    carrier_modulation::{bpsk_decode, bpsk_encode, null_decode, null_encode},
    correlation::{
        auto_correlate, cross_correlation_timing_metric,
        cross_correlation_timing_metric_single_value,
    },
    ofdm::{
        cross_correlation_to_known_signal, ofdm_frame_encoder, ofdm_preamble_encode,
        ofdm_premable_auto_correlation_detector, ofdm_premable_cross_correlation_detector,
        OfdmDataDecoder, OfdmDataEncoder, SubcarrierDecoder, SubcarrierEncoder,
    },
};
use num_complex::Complex;
use plotters::style::HSLColor;
use std::{io::Read, iter, path::Path};
use stft::{frequency_samples_to_time, time_samples_to_frequency};
const SEED: u64 = 0;
const REPEAT_CNT: usize = 10;

#[test]
fn scratch() -> anyhow::Result<()> {
    // Generate speaker amplitude iterator.
    // TODO don't leak.
    // Put on heap to avoid stack overflow.
    let subcarriers_encoders = Box::leak(Box::new(
        [SubcarrierEncoder::T0(null_encode::<f32>); time_samples_to_frequency(4800)],
    ));
    let subcarriers_decoders = Box::leak(Box::new(
        [SubcarrierDecoder::Data(null_decode); time_samples_to_frequency(4800)],
    ));
    // dbg!(subcarriers.len());
    // let first_bin =
    // (opt.start_freq / bin_width_from_freq(sample_rate as f32, 4800 / 2 + 1)) as usize;
    let first_bin = 20;
    let simultaneous_bytes = 8;
    let active_bins = first_bin..(first_bin + 8 * simultaneous_bytes);
    for i in active_bins.clone() {
        subcarriers_encoders[i] = SubcarrierEncoder::T1(bpsk_encode);
        subcarriers_decoders[i] = SubcarrierDecoder::Data(bpsk_decode);
    }

    // let mut ofdm = Ofdm::new(bits, subcarriers);
    let bytes_num = 256;
    let ofdm = OfdmDataEncoder::new(
        bytes_to_bits((0..=255).take(bytes_num)),
        &*subcarriers_encoders,
        480,
        1,
    );

    // Play sound -----------------------------------------------------------------
    /* let encoded_signal = ofdm_frame_encoder(SEED, REPEAT_CNT, ofdm.clone())
        // .map(|x| x.min(1.0))
        // .map(|x| x.max(-1.0))
        ;
    let normal = Normal::new(0.0, 1.0)?; // White additive noise channel
    let data = encoded_signal
        .clone()
        .map(|x| x as f64 / 10.0)
        // .map(|x| x + 0.05 * (normal.sample(&mut rand::thread_rng())))
        .collect::<Vec<_>>();
    {
        // Get duration of full tranmission.
        // let transmit_length = (ofdm.time_len() *  * 255 / 8;
        // let duration_of_transmission =
        // Duration::from_secs_f32(10.0 * transmit_length as f32 / sample_rate as f32);

        // Once partial symbols implemeneted below will work quicker.
        /* let data_duration = dbg!(
            ofdm.symbol_len() as f32 * ((bytes_num) as f32 / simultaneous_bytes as f32).ceil()
        ) / 48000f32; */
        let data_duration = dbg!(ofdm.clone().flatten().count() as f32) / 48000 as f32;
        let preamble_duration = dbg!(ofdm.time_len() + ofdm.symbol_len()) as f32 / 48000f32;
        let encoded_duration = dbg!(encoded_signal.clone().count());
        dbg!(encoded_duration as f32 - preamble_duration * 48000f32);
        let duration_of_transmission =
            std::time::Duration::from_secs_f32(data_duration + preamble_duration);

        // Play tranmission.
        let _stream = audio_network_interface::transmit::play_signal(encoded_signal)?;

        // write_wav(Path::new("out.wav"), encoded_signal.take(4800000))?;
        println!("Transmission of length {duration_of_transmission:?} playing.");
        // Stream will play until it is dropped.

        std::thread::sleep(duration_of_transmission);
        println!("Dropping played stream");
    } */
    // Play sound -----------------------------------------------------------------

    let (_spec, data) = read_wav(Path::new(r"out.wav"));
    let data = data
        .into_iter()
        // .map(|x| x * 100.0)
        .skip(20_000) // Skip startup samples
        .collect::<Vec<_>>();
    let tx_preamble = ofdm_preamble_encode(SEED, REPEAT_CNT, ofdm.time_len(), ofdm.cyclic_len())
        .collect::<Vec<_>>();
    let data_complex = data
        .clone()
        .into_iter()
        .map(|x| x / 2.0)
        .collect::<Vec<_>>();
    /* let auto_preamble_detect = dbg!(ofdm_premable_auto_correlation_detector(
        &data_complex,
        ofdm.time_len() / 2,
        0.75,
        ofdm.time_len() / 2
    ))
    .expect("Can't find packet.")
    .0 */
    let cross_preamble_detect = dbg!(ofdm_premable_cross_correlation_detector(
        &data_complex,
        &tx_preamble[..ofdm.time_len() / REPEAT_CNT],
        0.125,
    ))
    .expect("Can't find packet.")
    .0;

    {
        // Time Graph --------------------------------
        let cross = (0..data_complex.len() - tx_preamble.len()).map(|i| {
            cross_correlation_timing_metric_single_value(
                &data_complex[i..],
                &tx_preamble[..ofdm.time_len() / REPEAT_CNT],
                ofdm.time_len() / REPEAT_CNT,
            )
        });

        let cross_timing_metric = cross_correlation_timing_metric(
            &data_complex,
            &data_complex[ofdm.time_len() / REPEAT_CNT..],
            ofdm.time_len() / REPEAT_CNT,
        )
        .collect::<Vec<_>>();

        plot_series(
            "actual_wav_preamble.png",
            vec![
                data.clone(),
                cross.collect(),
                iter::repeat(0)
                    .enumerate()
                    .take(data.len())
                    .map(|(i, _)| {
                        let start = cross_preamble_detect;
                        if i < start {
                            0.0
                        } else if (i - start) % ofdm.symbol_len() == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                cross_correlation_to_known_signal(
                    &data_complex,
                    &tx_preamble[..ofdm.time_len() / REPEAT_CNT],
                )
                .collect(),
                /* cross_timing_metric.clone(),
                cross_timing_metric
                    .into_iter()
                    .rolling_average(ofdm.time_len())
                    .collect(), */
            ],
            "title",
            false,
            -0.5,
        )
        .unwrap();
    }

    // let rx_preamble = &data[cross_preamble_detect + ofdm.symbol_len()
    //     ..cross_preamble_detect + ofdm.symbol_len() + ofdm.time_len()];
    // let preamble_coefficients = scaled_real_fft(&mut rx_preamble.to_owned());

    // let rx_preamble2 = &data[cross_preamble_detect + ofdm.symbol_len() + ofdm.symbol_len()
    //     ..cross_preamble_detect + ofdm.symbol_len() + ofdm.time_len() + ofdm.symbol_len()];
    // let preamble_coefficients2 = scaled_real_fft(&mut rx_preamble2.to_owned());
    // dbg!(&preamble_coefficients);
    // dbg!(&rx_preamble);

    let decoder = OfdmDataDecoder::build(
        data.iter().copied().skip(cross_preamble_detect),
        *subcarriers_decoders,
        ofdm.cyclic_len(),
        SEED,
        REPEAT_CNT,
        1,
    );
    let mut decoder_c = decoder.clone();
    // dbg!((0..decoder.gain_factors().len())
    //     .map(|i| decoder.gain_factors()[i] * preamble_coefficients[i])
    //     .collect::<Vec<_>>());
    // dbg!((0..decoder.gain_factors().len())
    //     .map(|i| decoder.gain_factors()[i] * preamble_coefficients2[i])
    //     .collect::<Vec<_>>());
    // dbg!(tx_preamble.skip(ofdm.time_len()).collect::<Vec<_>>());
    // dbg!(decoder.gain_factors());

    let mut decoded = Vec::new();
    for data in decoder {
        decoded.push(data);
    }
    let mut decoded_bytes: BitVec<usize, Lsb0> = BitVec::new();
    decoded_bytes.extend(decoded.into_iter().flatten());
    let bytes = decoded_bytes.bytes().take(bytes_num);
    eprintln!("Decoded bytes: {:?}", bytes.collect::<Result<Vec<_>, _>>()?);

    let mut points = Vec::new();
    let gain_factors = decoder_c.gain_factors();
    const SKIP_CNT: usize = 0;
    iter::repeat_with(|| decoder_c.next_complex())
        .take(SKIP_CNT)
        .for_each(drop);
    for spectrum in iter::repeat_with(|| decoder_c.next_complex())
        .map_while(|x| x)
        .take(bytes_num / simultaneous_bytes)
        .enumerate()
    {
        let (idx, spectrum) = spectrum;
        let scaled_spectrum = spectrum
            .iter()
            .enumerate()
            .map(|(i, x)| *x * gain_factors[i]);
        let symbol_num = bytes_num / simultaneous_bytes;
        points.push(
            scaled_spectrum
                .enumerate()
                .filter(|(i, _)| active_bins.contains(&i))
                .map(|(_, x)| {
                    (
                        x.re as f32,
                        x.im as f32,
                        HSLColor(idx as f64 / (symbol_num as f64 * 1.2), 1.0, 0.5),
                    )
                })
                .collect::<Vec<_>>(),
        );
        /* points.push(
            spectrum
                .iter()
                .enumerate()
                .filter(|(i, _)| active_bins.contains(&i))
                .map(|(_, x)| {
                    (
                        x.re as f32,
                        x.im as f32,
                        HSLColor(idx as f64 / symbol_num as f64, 1.0, 0.1),
                    )
                })
                .collect::<Vec<_>>(),
        ); */
    }
    scatterplot(
        points
            .into_iter()
            .flatten()
            // .filter(|x| x.0.abs() <= 2.0 && x.1.abs() <= 2.0)
            .collect(),
        1,
        "scatterplot.png",
        "Complex spectrum",
    )?;

    Ok(())
}
