use log::info;
use num_traits::Zero;
use plotters::{prelude::*, style::SizeDesc};
use std::{cmp::PartialEq, path::Path};
use stft::Stft;

/// Plot a dtft analysis with proper scaling and labeling especially for the x axis frequency.
pub fn plot_fft(
    file_out: &Path,
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
    info!("Successfully saved fft graph of {}", file_out.display());
    Ok(())
}

/// Plot given series with x axis as each sample's index.
pub fn plot_series(
    file_out: &str,
    mut data: Vec<Vec<f64>>,
    title: &str,
    log_plot: bool,
    floor: f64,
) -> anyhow::Result<()> {
    for series in data.iter_mut() {
        for datapoint in series.iter_mut() {
            // log base 10 the data.
            if log_plot {
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
    info!("Successfully saved graph of series to {file_out}");
    Ok(())
}

/// Plot 2d spectogram.
pub fn plot_spectogram(
    stft: &Stft,
    bin_width: f32,
    file_out: &Path,
    title: &str,
    log_plot: bool,
) -> anyhow::Result<()> {
    let mut data = stft.data().to_owned();
    for series in data.iter_mut() {
        for datapoint in series.iter_mut() {
            // log base 10 the data.
            if log_plot {
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
    let root = BitMapBackend::new(&file_out, (1280, 720)).into_drawing_area();
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
    info!("Successfully saved fft graph of {}", file_out.display());
    Ok(())
}

/// Plot scatterplot.
pub fn scatterplot<S>(
    data: Vec<(f32, f32, S)>,
    radius: impl SizeDesc + Copy,
    file_out: &str,
    title: &str,
) -> anyhow::Result<()>
where
    S: Into<ShapeStyle> + Clone + PartialEq,
{
    // Find extents of the data.
    let mut max = (f32::zero(), f32::zero());
    let mut min = (f32::zero(), f32::zero());
    for point in data.iter() {
        max.0 = max.0.max(point.0);
        max.1 = max.1.max(point.1);
        min.0 = min.0.min(point.0);
        min.1 = min.1.min(point.1);
    }
    info!("Max of plot is {max:?}");
    info!("Min of plot is {min:?}");

    // Setup graph.
    let root = SVGBackend::new(&file_out, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(min.0..max.0, min.1..max.1)?;

    // Draw the tickmarks and mesh.
    chart
        .configure_mesh()
        // .disable_mesh()
        // .y_label_style(("sans-serif", 15).into_font())
        // .x_label_style(("sans-serif", 15).into_font())
        .x_desc("X Series")
        .y_desc("Y Series")
        .draw()?;

    // Plot all points.
    /* let plotting_area = chart.plotting_area();
    for (x, y, style) in data {
        plotting_area.draw_pixel((x, y), &style)?;
    } */
    let data = data.into_iter().collect::<Vec<_>>();
    let styles = {
        let mut out = data.clone();
        out.dedup_by_key(|x| x.2.clone());
        out.into_iter().map(|x| x.2)
    };
    chart.draw_series(
        data.into_iter()
            .map(|(x, y, style)| Circle::new((x, y), radius, style)),
    )?;
    // Add legend for each style
    styles.enumerate().for_each(|(i, style)| {
        chart
            // Don't actually draw anything
            .draw_series([Circle::new((0., 0.), 0, BLACK)])
            // but add to legend
            .expect("No fail on empty series")
            // label
            .label(i.to_string())
            // and color
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style.clone()));
    });

    // Draw legend
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    info!("Successfully saved fft graph of to {file_out}");
    Ok(())
}

/// Takes a length and a constant and generates an iterator that can be used for plotting with non constant data.
pub fn generate_constant_series<T: Clone>(
    constant: T,
    series_len: usize,
) -> impl Iterator<Item = T> {
    (0..series_len).map(move |_| constant.clone())
}
