use bitvec::{bitvec, prelude::Lsb0, vec::BitVec};
use iterator_adapters::IteratorAdapter;
use log::trace;
use ordered_float::OrderedFloat;
use std::io::Read;
use stft::Stft;

/// Sums the transient amplitude of each bin within the channel's frequency bounds.
pub fn channel_transient_amplitude(
    channel_lower: f32,
    channel_upper: f32,
    stft: &Stft,
    bin_width: f32,
) -> Vec<f64> {
    let mut transient = Vec::new();
    for (time_idx, time) in stft.times().enumerate() {
        transient.push(0.0);
        for (bin_idx, bin_value) in time.iter().enumerate() {
            let bin_center = bin_width * bin_idx as f32;
            // If the bin is in the channel add it to the channel's total
            if channel_lower <= bin_center && bin_center <= channel_upper {
                transient[time_idx] += *bin_value;
            }
        }
    }
    transient
}

/// Decodes an amplitude modulated audio signal.
#[derive(Debug, Default, Clone)]
pub struct AmplitudeModulationDecoder {
    pub bits_cnt: usize,
    pub frequency_channels: Vec<f32>,
    pub sample_rate: f32,
}

impl AmplitudeModulationDecoder {
    /// Decodes frequency analysis from [`stft::SpecCompute::stft()`] into data bits.
    /// Sensitivity is from 0 (most sensitive) to 1 (least sensitive).
    pub fn decode_amplitude_modulation(&self, stft: &Stft, sensitivity: f64) -> Vec<u8> {
        assert!(
            (0.0..=1.0).contains(&sensitivity),
            "Invalid sensitivity. Must be between 0.0 - 1.0"
        );
        let mut decoded = bitvec![0; self.bits_cnt];
        let frequency_channel_cnt = self.frequency_channels.len();
        // Ex. f 1 1 1
        //       1 0 1
        //       t
        // reads 1,0,1 of first frequency with f_idx = 0, v_idx = 0,1,2
        // so indices in self correspond to 0,2,4
        // next reads 1,1,1 of second frequency with f_idx = 1, v_idx = 0,1,2
        // so indices in self correspond to 1,3,5
        for (f_idx, frequency) in self.frequency_channels.iter().enumerate() {
            let bin = stft.get_bin(*frequency, self.sample_rate).unwrap();
            let threshold = bin.iter().copied().map(OrderedFloat).kth_order(90).0 / 2.0;
            let threshold = threshold * sensitivity;
            trace!("Threshold for frequency {frequency} Hz is {threshold}");
            for (v_idx, value) in bin.iter().enumerate() {
                if value > &threshold {
                    decoded.set(f_idx + v_idx * frequency_channel_cnt, true);
                }
            }
        }

        log::debug!("Bits before guard removed: {:?}", decoded);

        // When window first full of active bits the packet is started the bit after that window.
        let start_idx = decoded
            .windows(frequency_channel_cnt)
            .position(|x| {
                if x.iter().filter(|x| **x).count() == frequency_channel_cnt {
                    true
                } else {
                    false
                }
            })
            .expect("Can't find start.")
            + frequency_channel_cnt;
        // When window is first full of active bits in reverse the packet ends at the end of that window.
        let end_idx = decoded
            .windows(frequency_channel_cnt)
            .rev()
            // When window is full of active bits the packet is started.
            .position(|x| {
                if x.iter().filter(|x| **x).count() == frequency_channel_cnt {
                    true
                } else {
                    false
                }
            })
            .expect("Can't find end.")
            + frequency_channel_cnt;
        trace!(
            "Start at {start_idx}. End at {}",
            decoded.len() - 1 - end_idx
        );
        assert_ne!(
            start_idx,
            decoded.len() - 1 - end_idx,
            "Start idx is same as end idx"
        );

        // Remove stuff before and after guard symbol.
        let decoded: BitVec<u8, Lsb0> = decoded
            .into_iter()
            .skip(start_idx)
            .rev()
            .skip(end_idx)
            .rev()
            .collect();

        log::debug!("Bits after guard removed: {:?}", decoded);

        // Convert bit vector to byte vector.
        let bytes: Result<Vec<u8>, _> = decoded.bytes().collect();
        bytes.expect("Failed to convert decoded to bytes.")
    }
}
