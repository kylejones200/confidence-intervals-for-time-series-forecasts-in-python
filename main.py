#!/usr/bin/env python3
"""Confidence Intervals for Time Series Forecasts.

Bootstrap and parametric confidence intervals for ARIMA predictions.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src import (
    create_forecast_plot,
    ensure_output_dir,
    load_config,
    load_time_series,
)
from src.confidence_intervals import (
    bootstrap_confidence_intervals,
    parametric_confidence_intervals,
)
from src.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── data ──────────────────────────────────────────────────────────────────────

def _load_data(config: dict) -> tuple:
    """Load series and split into train/test."""
    series = load_time_series(
        config["data"]["input_file"],
        date_col=config["data"].get("date_col", "date"),
        value_col=config["data"].get("value_col", "value"),
    )
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    logger.info(f"Loaded {len(series)} points → train {len(train)}, test {len(test)}")
    return train, test, evaluator


def _forecast_index(train: pd.Series, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq=pd.infer_freq(train.index) or "D",
    )


# ── CI methods ────────────────────────────────────────────────────────────────

def _run_bootstrap_ci(config: dict, train, test, evaluator: Evaluator,
                      arima_order: tuple) -> None:
    """Fit ARIMA via bootstrap resampling and plot uncertainty bands."""
    n_bootstrap = config["model"].get("n_bootstrap", 100)
    alpha       = config["model"].get("alpha", 0.05)
    logger.info(f"Bootstrap CI (n={n_bootstrap})...")

    def fit_arima(data):
        return ARIMA(data, order=arima_order).fit()

    mean, lower, upper = bootstrap_confidence_intervals(
        model_fit_func=fit_arima,
        data=train,
        forecast_steps=len(test),
        n_bootstraps=n_bootstrap,
        confidence=1 - alpha,
        random_seed=config["model"].get("random_seed", 42),
    )
    idx      = _forecast_index(train, len(test))
    forecast = pd.Series(mean, index=idx)
    ci       = pd.DataFrame({"lower": lower, "upper": upper}, index=idx)

    aligned   = forecast.reindex(test.index, method="nearest")
    valid     = ~aligned.isna() & ~test.isna()
    metrics   = evaluator.evaluate(aligned[valid], test[valid]) if valid.sum() > 0 else {}
    rmse      = metrics.get("RMSE", 0)
    logger.info(f"Bootstrap RMSE: {rmse:.4f}")

    fig, _ = create_forecast_plot(
        train=train, test=test, forecast=forecast, conf_int=ci,
        figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
        title=f"Bootstrap Confidence Intervals (RMSE: {rmse:.4f})",
    )
    out = ensure_output_dir(config) / "bootstrap_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def _run_parametric_ci(config: dict, train, test, evaluator: Evaluator,
                       arima_order: tuple) -> None:
    """Fit ARIMA analytically and derive Gaussian prediction intervals."""
    alpha = config["model"].get("alpha", 0.05)
    logger.info("Parametric CI...")

    model             = ARIMA(train.values, order=arima_order).fit()
    mean, lower, upper = parametric_confidence_intervals(
        model=model, forecast_steps=len(test), confidence=1 - alpha,
    )
    idx      = _forecast_index(train, len(test))
    forecast = pd.Series(mean, index=idx)
    ci       = pd.DataFrame({"lower": lower, "upper": upper}, index=idx)

    aligned  = forecast.reindex(test.index, method="nearest")
    valid    = ~aligned.isna() & ~test.isna()
    metrics  = evaluator.evaluate(aligned[valid], test[valid]) if valid.sum() > 0 else {}
    rmse     = metrics.get("RMSE", 0)
    logger.info(f"Parametric RMSE: {rmse:.4f}")

    fig, _ = create_forecast_plot(
        train=train, test=test, forecast=forecast, conf_int=ci,
        figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
        title=f"Parametric Confidence Intervals (RMSE: {rmse:.4f})",
    )
    out = ensure_output_dir(config) / "parametric_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config      = load_config()
    train, test, evaluator = _load_data(config)
    arima_order = tuple(config["model"]["arima_order"])

    if config["model"].get("use_bootstrap", True):
        _run_bootstrap_ci(config, train, test, evaluator, arima_order)

    if config["model"].get("use_parametric", True):
        _run_parametric_ci(config, train, test, evaluator, arima_order)

    logger.info("Confidence interval analysis complete.")
    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
