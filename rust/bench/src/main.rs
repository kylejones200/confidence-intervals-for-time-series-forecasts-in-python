use confidence_intervals_for_time_series_forecasts_in_python_core::bootstrap_percentile_ci;

fn main() {
    let n_boot = 500usize;
    let horizon = 24usize;
    let paths: Vec<f64> = (0..n_boot * horizon).map(|i| (i as f64 * 0.01).sin()).collect();
    for _ in 0..500 {
        let _ = bootstrap_percentile_ci(&paths, n_boot, horizon, 0.05, 0.95);
    }
}
