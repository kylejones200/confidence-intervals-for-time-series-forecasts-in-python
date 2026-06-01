#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import bootstrap_percentile_ci  # noqa: E402

def main() -> None:
    nb, h = 500, 24
    paths = np.ascontiguousarray(np.sin(np.arange(nb * h) * 0.01))
    t0 = time.perf_counter()
    for _ in range(200):
        bootstrap_percentile_ci(paths, nb, h)
    py_s = time.perf_counter() - t0
    try:
        import confidence_intervals_for_time_series_forecasts_in_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(paths, nb, h, 0.05, 0.95, 500)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    pl, pu = bootstrap_percentile_ci(paths, nb, h)
    rl, ru = rs.bootstrap_percentile_ci_py(paths, nb, h, 0.05, 0.95)
    np.testing.assert_allclose(pl, np.asarray(rl), rtol=1e-10)
    np.testing.assert_allclose(pu, np.asarray(ru), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
