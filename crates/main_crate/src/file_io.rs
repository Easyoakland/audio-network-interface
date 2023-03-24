use std::{
    fs::File,
    io::{BufReader, BufWriter, Bytes, Read, Write},
    path::Path,
};

use hound::{SampleFormat, WavReader, WavSpec};

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
pub fn read_wav(file: &Path) -> (WavSpec, Vec<f64>) {
    // The WAV file to decode.
    let mut reader =
        WavReader::open(file).unwrap_or_else(|err| panic!("Invalid wav file {file:?}: {err}."));
    let spec = reader.spec();
    log::info!("Spec: {:?}", reader.spec());
    // Select correct format representation.
    let data = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .step_by(spec.channels.into()) // Make wav mono for analysis. Ignore all but first channel.
            .map(|x| x.unwrap_or_else(|err| panic!("Error reading sample: {err}")) as f64)
            .collect(),
        SampleFormat::Int => reader
            .samples::<i32>()
            .step_by(spec.channels.into()) // Make wav mono for analysis. Ignore all but first channel.
            .map(|x| x.unwrap_or_else(|err| panic!("Error reading sample: {err}")) as f64)
            .collect(),
    };
    (spec, data)
}
