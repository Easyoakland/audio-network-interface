use log::info;
use plotters::prelude::*;
use stft::Stft;

/// Plot a dtft analysis with proper scaling and labeling especially for the x axis frequency.
pub fn plot_fft(
    file_in: &str,
    file_out: &str,
    data: Vec<f64>,
    sample_rate: f32,
    bin_width: f32,
) -> anyhow::Result<()> {
    // log base 10 the data. Floor of -1.
    let data = data
        .into_iter()
        .map(|x| x.log10())
        .map(|x| x.max(-1f64))
        .collect::<Vec<_>>();

    // Find max and min of data
    let max = *data
        .iter()
        .max_by(|&&x, &y| x.partial_cmp(y).unwrap())
        .unwrap();
    let min = *data
        .iter()
        .min_by(|&&x, &y| x.partial_cmp(y).unwrap())
        .unwrap();
    info!("Max of plot is {max}");
    info!("Min of plot is {min}");

    // setup graph
    let root = BitMapBackend::new(&file_out, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Relative amplitude per frequency",
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..(sample_rate / 2f32 + 1f32), min..max)?;

    // draw the tickmarks and mesh
    chart
        .configure_mesh()
        .y_label_style(("sans-serif", 15).into_font())
        .x_label_style(("sans-serif", 15).into_font())
        .x_desc("Frequency in Hz")
        .y_desc("Relative Amplitude after log_10.")
        .draw()?;

    // generate pairs of bin positions and values
    let series_point_iter = data
        .into_iter()
        .enumerate()
        .map(|(i, x)| ((i as f32) * bin_width, x));

    // draw series
    chart.draw_series(LineSeries::new(series_point_iter, &RED))?;

    root.present()?;
    info!("Successfully saved fft graph of \"{file_in}\" to {file_out}");
    Ok(())
}

/// Plot given series with x axis as each sample's index.
pub fn plot_series(
    file_in: &str,
    file_out: &str,
    mut data: Vec<Vec<f64>>,
    title: &str,
    should_log: bool,
    floor: f64,
) -> anyhow::Result<()> {
    for series in data.iter_mut() {
        for datapoint in series.iter_mut() {
            // log base 10 the data.
            if should_log {
                *datapoint = datapoint.log10();
            }
            // Floor data.
            *datapoint = datapoint.max(floor);
        }
    }

    // Find max and min of data
    let mut max = f64::NAN;
    let mut min = f64::NAN;
    for series in data.iter() {
        for datapoint in series {
            max = datapoint.max(max);
            min = datapoint.min(min);
        }
    }
    info!("Max of plot is {max}");
    info!("Min of plot is {min}");

    // setup graph
    let root = BitMapBackend::new(&file_out, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(40)
        .build_cartesian_2d(0..(data[0].len()), min..max)?;

    // draw the tickmarks and mesh
    chart
        .configure_mesh()
        .y_label_style(("sans-serif", 15).into_font())
        .x_label_style(("sans-serif", 15).into_font())
        .x_desc("X series")
        .y_desc("Y series")
        .draw()?;

    // draw series
    let data_len = data.len();
    for (i, series) in data.into_iter().enumerate() {
        chart.draw_series(LineSeries::new(
            series.into_iter().enumerate(),
            &HSLColor(i as f64 / data_len as f64, 1.00, 0.40),
        ))?;
    }

    root.present()?;
    info!("Successfully saved fft graph of \"{file_in}\" to {file_out}");
    Ok(())
}

/// Plot 2d spectogram.
pub fn plot_spectogram(
    stft: &Stft,
    bin_width: f32,
    file_in: &str,
    file_out: &str,
    title: &str,
    should_log: bool,
) -> anyhow::Result<()> {
    let mut data = stft.data().to_owned();
    for series in data.iter_mut() {
        for datapoint in series.iter_mut() {
            // log base 10 the data.
            if should_log {
                *datapoint = datapoint.log10();
            }
            //  Floor of -1.
            *datapoint = datapoint.max(-1f64);
        }
    }

    // Find min and max frequencies given.
    let high_freq = bin_width * data.len() as f32;
    let low_freq = 0.0;

    // Find max and min of data
    let mut max = 0.0;
    let mut min = 0.0;
    for transient in data.iter() {
        for datapoint in transient {
            max = datapoint.max(max);
            min = datapoint.min(min);
        }
    }
    info!("Max of plot is {max}");
    info!("Min of plot is {min}");

    // setup graph
    let root = BitMapBackend::new(
        &file_out,
        (data[0].len().min(1280usize).try_into().unwrap(), 720),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..data[0].len(), low_freq..high_freq)?;

    // draw the tickmarks and mesh
    chart
        .configure_mesh()
        .disable_mesh()
        // .y_label_style(("sans-serif", 15).into_font())
        // .x_label_style(("sans-serif", 15).into_font())
        .x_desc("X window")
        .y_desc("Frequency (Hz)")
        .draw()?;

    let plotting_area = chart.plotting_area();

    let range = plotting_area.get_pixel_range();
    let (_width_px, height_px) = (range.0.end - range.0.start, range.1.end - range.1.start);
    let height_value = chart.y_range().end - chart.y_range().start;
    let (x_range, y_range) = (
        chart.x_range(),
        chart.y_range().step(height_value / height_px as f32), // step by a pixels worth of value to minimize work
    );

    // Draw spectogram.
    for x in x_range {
        // For time
        for y in y_range.values() {
            let color = &HSLColor(data[(y / bin_width) as usize][x] / max, 1.0, 0.5);
            // Don't draw red everywhere if there is a near-zero value.
            if !(color.0 < 0.01) {
                // For frequency
                plotting_area.draw_pixel((x, y), color)?;
            }
        }
    }

    root.present()?;
    info!("Successfully saved fft graph of \"{file_in}\" to {file_out}");
    Ok(())
}

/// Takes a length and a constant and generates an iterator that can be used for plotting with non constant data.
pub fn generate_constant_series<T: Clone>(
    constant: T,
    series_len: usize,
) -> impl Iterator<Item = T> {
    (0..series_len).map(move |_| constant.clone())
}
