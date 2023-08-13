use anyhow::Context;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
#[cfg(target_arch = "wasm32")]
use log::info;
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
};
use std::{
    io::{self, Cursor},
    path::Path,
};

/// Read file to byte iterator.
pub async fn read_file_bytes(
    file: &Path,
) -> io::Result<impl Iterator<Item = Result<u8, io::Error>> + core::fmt::Debug> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file_handle = File::open(file)?;
        let reader = io::BufReader::new(file_handle);
        Ok(reader.bytes())
    }
    // Can't read file paths normally in wasm. Instead create an async dialogue that reads a file to memory.
    #[cfg(target_arch = "wasm32")]
    {
        Ok(rfd::AsyncFileDialog::new()
            .set_title(&file.display().to_string())
            .pick_file()
            .await
            .ok_or(io::Error::new(
                io::ErrorKind::Interrupted,
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
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file_handle = File::create(file)?;
        let mut writer = BufWriter::new(file_handle);
        writer.write_all(data)?;
    }
    // TODO write to file instead of logging
    #[cfg(target_arch = "wasm32")]
    {
        info!(
            "Writing to '{}': {:?}",
            file.display(),
            data.iter().map(|&x| x as char).collect::<String>()
        )
    }
    Ok(())
}

/// Read data from a wav file.
pub async fn read_wav(file: &Path) -> anyhow::Result<(WavSpec, Vec<f64>)> {
    let bytes = read_file_bytes(file)
        .await
        .with_context(|| format!("reading file {}", file.display()))?
        .collect::<Result<Vec<_>, io::Error>>()?;
    // The WAV file to decode.
    let mut reader =
        WavReader::new(&*bytes).with_context(|| format!("Invalid wav file {}", file.display()))?;
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
pub fn write_wav(file: &Path, samples: impl Iterator<Item = f32>) -> anyhow::Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut memory_writer = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut memory_writer, spec)?;
    for sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    write_file_bytes(file, &memory_writer.into_inner())?;
    Ok(())
}
