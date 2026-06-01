//! Bootstrap percentile confidence bands from resampled forecast paths.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 as usize) % upper
    }
}

/// Returns (lower, upper) per horizon from bootstrap draws (row-major n_boot x horizon).
pub fn bootstrap_percentile_ci(
    paths: &[f64],
    n_boot: usize,
    horizon: usize,
    lower_q: f64,
    upper_q: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(paths.len(), n_boot * horizon);
    let mut lower = Vec::with_capacity(horizon);
    let mut upper = Vec::with_capacity(horizon);
    for h in 0..horizon {
        let mut col: Vec<f64> = (0..n_boot).map(|b| paths[b * horizon + h]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let li = ((n_boot as f64) * lower_q).floor() as usize;
        let ui = ((n_boot as f64) * upper_q).ceil() as usize;
        let li = li.min(n_boot.saturating_sub(1));
        let ui = ui.min(n_boot.saturating_sub(1)).max(li);
        lower.push(col[li]);
        upper.push(col[ui]);
    }
    (lower, upper)
}

/// Draw bootstrap paths by resampling residuals and cumulating (simplified demo kernel).
pub fn bootstrap_forecast_paths(
    history: &[f64],
    residuals: &[f64],
    horizon: usize,
    n_boot: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n_boot * horizon);
    let last = *history.last().unwrap_or(&0.0);
    for _ in 0..n_boot {
        let mut level = last;
        for _ in 0..horizon {
            let r = residuals[rng.next_usize(residuals.len())];
            level += r;
            out.push(level);
        }
    }
    out
}
