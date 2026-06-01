use confidence_intervals_for_time_series_forecasts_in_python_core::bootstrap_percentile_ci;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn bootstrap_percentile_ci_py<'py>(
    py: Python<'py>,
    paths: PyReadonlyArray1<f64>,
    n_boot: usize,
    horizon: usize,
    lower_q: f64,
    upper_q: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (lo, hi) = bootstrap_percentile_ci(paths.as_slice()?, n_boot, horizon, lower_q, upper_q);
    Ok((lo.into_pyarray(py), hi.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (paths, n_boot, horizon, lower_q=0.05, upper_q=0.95, iterations=500))]
fn bench_kernel_py(
    paths: PyReadonlyArray1<f64>,
    n_boot: usize,
    horizon: usize,
    lower_q: f64,
    upper_q: f64,
    iterations: usize,
) -> PyResult<f64> {
    let buf = paths.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = bootstrap_percentile_ci(&buf, n_boot, horizon, lower_q, upper_q);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn confidence_intervals_for_time_series_forecasts_in_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap_percentile_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
