use anyhow::Context;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Bytes, Read, Write},
    path::Path,
};

/// Read file to byte iterator.
pub fn read_file_bytes(file: &Path) -> anyhow::Result<Bytes<BufReader<File>>> {
    let file_handle = File::open(file)?;
    let reader = BufReader::new(file_handle);
    Ok(reader.bytes())
}

/// Write byte slice to specified file.
pub fn write_file_bytes(file: &Path, data: &[u8]) -> anyhow::Result<()> {
    let file_handle = File::create(file)?;
    let mut writer = BufWriter::new(file_handle);
    writer.write_all(data)?;
    Ok(())
}

/// Read data from a wav file.
pub fn read_wav(file: &Path) -> anyhow::Result<(WavSpec, Vec<f64>)> {
    // The WAV file to decode.
    let mut reader = WavReader::open(file).context("Invalid wav file")?;
    let spec = reader.spec();
    log::trace!("Spec: {:?}", reader.spec());
    // Select correct format representation.
    let data = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .step_by(spec.channels.into()) // Make wav mono for analysis. Ignore all but first channel.
            .map(|x| f64::from(x.unwrap_or_else(|err| panic!("Error reading sample: {err}"))))
            .collect(),
        SampleFormat::Int => reader
            .samples::<i32>()
            .step_by(spec.channels.into()) // Make wav mono for analysis. Ignore all but first channel.
            .map(|x| f64::from(x.unwrap_or_else(|err| panic!("Error reading sample: {err}"))))
            .collect(),
    };
    Ok((spec, data))
}

/// Write data to wav file.
pub fn write_wav(file: &Path, samples: impl Iterator<Item = f32>) -> Result<(), hound::Error> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(file, spec)?;
    for sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}
