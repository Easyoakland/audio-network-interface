use anyhow::Context;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Read file to byte iterator.
pub async fn read_file_bytes(
    file: &Path,
) -> anyhow::Result<impl Iterator<Item = Result<u8, std::io::Error>> + core::fmt::Debug> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file_handle = File::open(file)?;
        let reader = std::io::BufReader::new(file_handle);
        Ok(reader.bytes())
    }
    // Can't read file paths normally in wasm. Instead create an async dialogue that reads a file to memory.
    #[cfg(target_arch = "wasm32")]
    {
        Ok(rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                format!("File dialogue closed while picking \"{}\"", file.display()),
            ))?
            .read()
            .await
            .into_iter()
            .map(Ok))
    }
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
    let mut reader =
        WavReader::open(file).with_context(|| format!("Invalid wav file {}", file.display()))?;
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
