# Confidence Intervals for Time Series Forecasts with Python

All forecasts contain uncertainty. Point predictions provide the best estimate of future values and confidence intervals help quantify the range of likely outcomes. As we make predictions further into the future, uncertainty increases due to:

- **Model Uncertainty:** Inaccuracies inherent in the model's assumptions and structure.

- **Parameter Uncertainty:** Variability in estimated parameters, impacting model predictions.

- **Randomness:** Inherent randomness in the system being modeled.

Confidence intervals quantify this uncertainty and allow us to communicate the range of potential outcomes clearly and effectively.

# Computing Forecast Confidence Intervals

We start with a practical example using ARIMA models, which provide built-in confidence interval calculations. The example uses data from ERCOT on electricity demand, reported every 15 minutes. For easier analysis, the data is resampled to an hourly frequency.

## Data Loading and Preprocessing

from statsmodels.tsa.arima.model import ARIMA from sklearn.preprocessing import StandardScaler from pmdarima import auto_arima

import warnings warnings.filterwarnings("ignore")

    # Load and preprocess data
def load_and_preprocess_data(url): df = pd.read_csv(url) df['date'] = pd.to_datetime(df['date']) df.set_index('date', inplace=True) df = df.resample('h').mean().asfreq('h') df['values'] = df['values'].interpolate()

scaler = StandardScaler() df['scaled_values'] = scaler.fit_transform(df[['values']])

return df, scaler

## ARIMA Forecast with Confidence Intervals

ARIMA models are widely used in time series forecasting due to their flexibility in handling different patterns, including trend and seasonality. They also provide built-in confidence intervals for forecasts.

    # Forecast with ARIMA
def forecast_with_confidence(data, order, steps=48, confidence=0.95): model = ARIMA(data, order=order) fitted_model = model.fit()

forecast_result = fitted_model.get_forecast(steps=steps) forecasts = forecast_result.predicted_mean conf_int = forecast_result.conf_int(alpha=1 - confidence)

return forecasts, conf_int.iloc[:, 0], conf_int.iloc[:, 1]

## Plotting Forecasts with Confidence Intervals

The forecast, along with its confidence intervals, is visualized using Matplotlib:

    # Plot function
def plot_forecast_with_ci(historical_data, test_data, forecasts, lower_ci, upper_ci, title="Forecast with Confidence Intervals"): plt.figure(figsize=(12, 6)) plt.plot(historical_data.index, historical_data.values, label='Historical Data', color='blue') plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')

forecast_index = test_data.index plt.plot(forecast_index, forecasts, 'r-', label='Forecast') plt.fill_between(forecast_index, lower_ci, upper_ci, color='r', alpha=0.2, label='95% CI')

plt.axvline(x=test_data.index[0], color='black', linestyle='--', label="Test Data Start") plt.title(title) plt.xlabel('Date') plt.ylabel('Value') plt.legend() plt.xticks(rotation=45) plt.tight_layout() plt.savefig(f'{title}.png') plt.show()

## Main Workflow

    # Main workflow
url = "https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main/ercot_load_data.csv" df, scaler = load_and_preprocess_data(url)

train_data = df['scaled_values'].iloc[:-48] test_data = df['scaled_values'].iloc[-48:]

    # Find best ARIMA order
auto_model = auto_arima(train_data, seasonal=False, trace=True, suppress_warnings=True, stepwise=True) best_order = auto_model.order print(f"Using ARIMA order: {best_order}")

    # ARIMA forecast with confidence intervals
forecasts, lower_ci, upper_ci = forecast_with_confidence(train_data, best_order, steps=48)

# Bootstrapped Confidence Intervals

For models that don't provide built-in confidence intervals, bootstrap methods can be used to estimate prediction intervals. Bootstrapping generates multiple samples by resampling the original data with replacement, fitting the model on each sample, and computing the forecast distribution.

## Bootstrap-based Forecast Confidence Intervals

    # Bootstrap-based forecast confidence intervals
def bootstrap_forecast_ci(model_order, data, steps=48, n_bootstraps=100, confidence=0.95): forecasts = []

for i in range(n_bootstraps): try: bootstrap_sample = data.sample(n=len(data), replace=True).sort_index() model = ARIMA(bootstrap_sample, order=model_order) fitted_model = model.fit() forecasts.append(fitted_model.forecast(steps=steps).values) except Exception as e: print(f"Bootstrap iteration {i} failed: {e}")

if not forecasts: raise RuntimeError("All bootstrap iterations failed.")

forecasts = np.array(forecasts) lower_ci = np.percentile(forecasts, (1 - confidence) / 2 * 100, axis=0) upper_ci = np.percentile(forecasts, (1 + confidence) / 2 * 100, axis=0) mean_forecast = np.mean(forecasts, axis=0)

return mean_forecast, lower_ci, upper_ci

## Bootstrapped Confidence Interval Plot

    # Bootstrapped confidence intervals
boot_forecasts, boot_lower_ci, boot_upper_ci = bootstrap_forecast_ci(best_order, train_data, steps=48, n_bootstraps=50)

    # Plot results
plot_forecast_with_ci(df['values'], test_data_original_series, boot_forecasts, boot_lower_ci, boot_upper_ci, title="Bootstrapped Forecast with Confidence Intervals")

# Interpreting Confidence Bounds

Confidence intervals require careful interpretation:

- A 95% confidence interval does not mean there is a 95% probability that the true value will fall within the interval.

- Instead, it means that if the process were repeated many times, about 95% of the calculated intervals would contain the true value.

- Wider intervals indicate greater uncertainty, while narrower intervals suggest more precision.

# Best Practices for Using Confidence Intervals

- **Contextual Interpretation:** Interpret confidence intervals in the context of the dataset and application.

- **Assumptions and Limitations:** Clearly communicate underlying assumptions (e.g., normality, stationarity).

- **Visualization:** Use effective visualizations (e.g., shaded regions, error bars) to make uncertainty intuitive and accessible.

- **Multiple Confidence Levels:** Consider multiple confidence levels (e.g., 80% and 95%) for a nuanced view of uncertainty.

- **Ongoing Validation:** Regularly validate confidence intervals against out-of-sample data to ensure well-calibrated predictions.

Confidence intervals enhance the interpretability and reliability of time series forecasts by quantifying uncertainty. Whether using ARIMA's built-in intervals or bootstrapped methods, confidence intervals provide a range of likely outcomes, enabling better decision-making and risk management.

By understanding and communicating the assumptions behind confidence intervals and using them in conjunction with effective visualizations, forecasters can present more transparent and actionable insights. Ongoing validation and consideration of multiple confidence levels further strengthen the credibility of predictions.

## Key Takeaways

- **Model Uncertainty:** Inaccuracies inherent in the model's assumptions and structure.
- **Parameter Uncertainty:** Variability in estimated parameters, impacting model predictions.
- **Randomness:** Inherent randomness in the system being modeled.
- A 95% confidence interval does not mean there is a 95% probability that the true value will fall within the interval.
