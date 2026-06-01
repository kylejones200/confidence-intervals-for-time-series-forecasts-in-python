"""Bootstrap percentile confidence intervals per forecast horizon."""

from __future__ import annotations

import numpy as np


def bootstrap_percentile_ci(
    paths: np.ndarray,
    n_boot: int,
    horizon: int,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(paths, dtype=float).reshape(n_boot, horizon)
    lower = np.percentile(p, lower_q * 100, axis=0)
    upper = np.percentile(p, upper_q * 100, axis=0)
    return lower, upper
