#!/usr/bin/env python3
"""
Confidence Intervals for Time Series Forecasts
Bootstrap and parametric confidence intervals for time series predictions.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Add src to path

import matplotlib.pyplot as plt
import pandas as pd

# Import consolidated utilities (signalplot already applied in src/__init__.py)
from src import (
    create_forecast_plot,
    ensure_output_dir,
    get_output_dir,
    load_config,
    load_time_series,
    save_plot,
)
from src.confidence_intervals import (
    bootstrap_confidence_intervals,
    parametric_confidence_intervals,
)
from src.evaluator import Evaluator
from statsmodels.tsa.arima.model import ARIMA


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent

    # Load configuration using consolidated loader
    config = load_config()

    # Load data using consolidated loader
    series = load_time_series(
        config["data"]["input_file"],
        date_column=config["data"].get("date_col", "date"),
        value_column=config["data"].get("value_col", "value"),
    )

    logger.info(f"Loaded {len(series)} data points")

    # Split train/test using consolidated evaluator
    evaluator = Evaluator(test_size=config.get("evaluation", {}).get("test_size", 0.2))
    train, test = evaluator.split(series)
    logger.info(f"Train: {len(train)} points, Test: {len(test)} points")

    # Model parameters
    arima_order = tuple(config["model"]["arima_order"])
    forecast_steps = len(test)
    alpha = config["model"].get("alpha", 0.05)
    n_bootstrap = config["model"].get("n_bootstrap", 100)

    # Generate bootstrap CI using consolidated utility
    if config["model"].get("use_bootstrap", True):
        logger.info(f"\nGenerating bootstrap confidence intervals (n={n_bootstrap})...")

        # Create model fit function
        def fit_arima(data):
            return ARIMA(data, order=arima_order).fit()

        mean_bs, lower_bs, upper_bs = bootstrap_confidence_intervals(
            model_fit_func=fit_arima,
            data=train,
            forecast_steps=forecast_steps,
            n_bootstraps=n_bootstrap,
            confidence=1 - alpha,
            random_seed=config["model"].get("random_seed", 42),
        )

        # Create forecast index
        forecast_index = pd.date_range(
            start=train.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq=pd.infer_freq(train.index) or "D",
        )

        bootstrap_forecast = pd.Series(mean_bs, index=forecast_index)
        bootstrap_ci_df = pd.DataFrame(
            {"lower": lower_bs, "upper": upper_bs}, index=forecast_index
        )

        # Evaluate
        bootstrap_aligned = bootstrap_forecast.reindex(test.index, method="nearest")
        valid_idx = ~bootstrap_aligned.isna() & ~test.isna()
        if valid_idx.sum() > 0:
            metrics = evaluator.evaluate(bootstrap_aligned[valid_idx], test[valid_idx])
            logger.info(f"Bootstrap Forecast - RMSE: {metrics['RMSE']:.4f}")

        # Plot
        fig, ax = create_forecast_plot(
            train=train,
            test=test,
            forecast=bootstrap_forecast,
            conf_int=bootstrap_ci_df,
            figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
            title=f"Bootstrap Confidence Intervals (RMSE: {metrics.get('RMSE', 0):.4f})",
        )

        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "bootstrap_ci.png", dpi=300)
        logger.info(f"Plot saved to: {output_dir / 'bootstrap_ci.png'}")

    # Generate parametric CI using consolidated utility
    if config["model"].get("use_parametric", True):
        logger.info("\nGenerating parametric confidence intervals...")

        model = ARIMA(train.values, order=arima_order).fit()
        mean_param, lower_param, upper_param = parametric_confidence_intervals(
            model=model,
            forecast_steps=forecast_steps,
            confidence=1 - alpha,
        )

        # Create forecast index
        forecast_index = pd.date_range(
            start=train.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq=pd.infer_freq(train.index) or "D",
        )

        parametric_forecast = pd.Series(mean_param, index=forecast_index)
        parametric_ci_df = pd.DataFrame(
            {"lower": lower_param, "upper": upper_param}, index=forecast_index
        )

        # Evaluate
        parametric_aligned = parametric_forecast.reindex(test.index, method="nearest")
        valid_idx = ~parametric_aligned.isna() & ~test.isna()
        if valid_idx.sum() > 0:
            metrics = evaluator.evaluate(parametric_aligned[valid_idx], test[valid_idx])
            logger.info(f"Parametric Forecast - RMSE: {metrics['RMSE']:.4f}")

        # Plot
        fig, ax = create_forecast_plot(
            train=train,
            test=test,
            forecast=parametric_forecast,
            conf_int=parametric_ci_df,
            figsize=tuple(config.get("plotting", {}).get("figure_size", [12, 6])),
            title=f"Parametric Confidence Intervals (RMSE: {metrics.get('RMSE', 0):.4f})",
        )

        output_dir = ensure_output_dir(get_output_dir(config, script_dir))
        save_plot(fig, output_dir / "parametric_ci.png", dpi=300)
        logger.info(f"Plot saved to: {output_dir / 'parametric_ci.png'}")

    logger.info("\n Confidence interval analysis complete")

    if config.get("plotting", {}).get("show_plot", True):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
