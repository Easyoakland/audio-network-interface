#![cfg(feature = "plot")]
use audio_network_interface::{
    file_io::read_wav,
    plotting::{plot_series, scatterplot},
};
use dsp::{
    carrier_modulation::{bpsk_decode, null_decode},
    ofdm::{
        cross_correlation_to_known_signal, ofdm_preamble_encode,
        ofdm_premable_cross_correlation_detector, OfdmFramesDecoder, SubcarrierDecoder,
    },
    specs::OfdmSpec,
};
use futures_lite::future::block_on;
use plotters::style::HSLColor;
use std::{env::args, iter, path::Path};
use stft::time_samples_to_frequency;
const REPEAT_CNT: usize = 10;
const BINS: usize = 8;
fn active_bins(first_bin: usize) -> std::ops::Range<usize> {
    first_bin..(first_bin + BINS)
}
// static SUBCARRIER_ENCODERS: Box<[SubcarrierEncoder<bool, num_complex::Complex<f32>>; 2401]> =
//     Box::new([SubcarrierEncoder::T0(null_encode::<f32>); time_samples_to_frequency(4800)]);
// static SUBCARRIER_DECODERS: Box<[SubcarrierDecoder<'_, f64>; 2401]> =
//     Box::new([SubcarrierDecoder::Data(null_decode::<f64>); time_samples_to_frequency(4800)]);

fn cross_correlation_plot(ofdm_spec: &OfdmSpec, data: &[f64]) -> anyhow::Result<()> {
    let tx_preamble = ofdm_spec.preamble().collect::<Vec<_>>();
    let cross_preamble_detect = ofdm_premable_cross_correlation_detector(
        &data,
        &tx_preamble[..ofdm_spec.time_symbol_len / REPEAT_CNT],
        ofdm_spec.cross_correlation_threshold.into(),
    )
    .map(|x| x.0);

    let per_symbol_marking = cross_preamble_detect
        .map(|start| {
            iter::repeat(0)
                .enumerate()
                .take(data.len())
                .map(|(i, _)| {
                    if i < start {
                        0.0
                    } else if (i - start) % ofdm_spec.symbol_len() == 0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let frame_len = ofdm_spec.frame_len();
    let per_frame_marking = cross_preamble_detect
        .map(|start| {
            iter::repeat(0)
                .enumerate()
                .take(data.len())
                .map(|(i, _)| {
                    if i < start {
                        0.0
                    } else if (i - start) % frame_len == 0 {
                        1.1
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let cross_correlation = cross_correlation_to_known_signal(
        &data,
        &tx_preamble[..ofdm_spec.time_symbol_len / REPEAT_CNT],
    )
    .collect();

    plot_series(
        "cross_correlation.png",
        vec![
            data.to_owned(),
            cross_correlation,
            per_symbol_marking,
            per_frame_marking,
        ],
        "cross correlation",
        false,
        -0.5,
    )?;

    Ok(())
}

fn iq_plane(
    ofdm_spec: &OfdmSpec,
    data: &[f64],
    subcarrier_decoders: [SubcarrierDecoder<'_, f64>; 2401],
) -> anyhow::Result<()> {
    let mut decoder = OfdmFramesDecoder::new(
        data.into_iter().copied(),
        subcarrier_decoders,
        ofdm_spec.clone(),
    );
    let frames = decoder.clone().count();
    let mut points = Vec::new();
    for (idx, (_gain_factors, scaled_spectrum_per_time)) in
        iter::repeat_with(|| decoder.next_frame_complex())
            .map_while(|x| x)
            .enumerate()
    {
        for scaled_spectrum in scaled_spectrum_per_time {
            points.push(
                scaled_spectrum
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| active_bins(ofdm_spec.first_bin).contains(&i))
                    .map(|(_, x)| {
                        (
                            x.re as f32,
                            x.im as f32,
                            HSLColor(idx as f64 / frames as f64, 1.0, 0.5),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
    scatterplot(
        points.into_iter().flatten().collect(),
        1,
        "scatterplot.svg",
        "IQ Plane colorized by frame ordered in time",
    )?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let ofdm_spec: OfdmSpec = OfdmSpec {
        ..Default::default()
    };
    let args = args().collect::<Vec<_>>();

    let path = Path::new(
        args.iter()
            .skip_while(|x| x != &"-p" && x != &"--path")
            .skip(1)
            .next()
            .map(|x| dbg!(x))
            .expect("path"),
    );
    let (_spec, data) = block_on(read_wav(path))?;
    let data = &data[0..];
    let tx_preamble = ofdm_preamble_encode(&ofdm_spec).collect::<Vec<_>>();
    let cross_preamble_detect = ofdm_premable_cross_correlation_detector(
        &data,
        &tx_preamble[..ofdm_spec.time_symbol_len / REPEAT_CNT],
        ofdm_spec.cross_correlation_threshold.into(),
    )
    .map(|x| x.0);
    let data_after_start = cross_preamble_detect
        .map(|start| &data[start..])
        .unwrap_or(data);

    cross_correlation_plot(&ofdm_spec, data)?;
    dbg!(cross_preamble_detect);
    let subcarrier_decoders = {
        let mut out = Box::new(
            [SubcarrierDecoder::Data(null_decode::<f64>); time_samples_to_frequency(4800)],
        );
        for bin in active_bins(ofdm_spec.first_bin) {
            out[bin] = SubcarrierDecoder::Data(bpsk_decode);
        }
        out
    };

    iq_plane(&ofdm_spec, data_after_start, *subcarrier_decoders)?;

    Ok(())
}
