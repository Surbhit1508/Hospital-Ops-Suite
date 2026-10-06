"""
Forecasting: weekly hospital admission volume, backtested honestly.

Data: real HHS/CDC weekly COVID-19 hospitalization metrics, national
aggregate ("USA" row), 2020-08-08 through the report's last update. This
is historical/deprecated reporting -- we use it to DEMONSTRATE a rigorous
forecasting methodology (walk-forward backtesting + prediction intervals),
not to claim a live production forecast of anything happening today.

Rigor: instead of one lucky train/test split, we do expanding-window
walk-forward backtesting -- repeatedly train on everything up to week t,
forecast the next `step` weeks, slide forward, and pool the errors. This
is the honest way to evaluate a time series model (a single split can
make a bad model look great or a good model look terrible by luck).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src import config
from src.stats.bootstrap import bootstrap_ci, paired_bootstrap_test

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message="Optimization failed to converge")


def load_national_series() -> pd.Series:
    df = pd.read_csv(config.WEEKLY_HOSPITALIZATIONS)
    df = df[df["state"] == config.FORECAST_STATE].copy()
    df["week_ending_date"] = pd.to_datetime(df["week_ending_date"])
    df = df.sort_values("week_ending_date")
    series = df.set_index("week_ending_date")[config.FORECAST_TARGET_COLUMN]
    series = series.asfreq("W-SAT")  # real data's week_ending_date always falls on Saturday
    series = series.interpolate(limit_direction="both")
    return series


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, history[-1])


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = 52) -> np.ndarray:
    if len(history) < season:
        return naive_forecast(history, horizon)
    return np.array([history[-season + (i % season)] for i in range(horizon)])


def holt_winters_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(history, trend="add", damped_trend=True, seasonal=None)
    fitted = model.fit(optimized=True)
    return fitted.forecast(horizon)


FORECASTERS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "holt_winters": holt_winters_forecast,
}


def walk_forward_backtest(series: pd.Series) -> dict[str, list[float]]:
    """Expanding-window backtest. Returns per-method lists of absolute errors
    pooled across every backtest origin and every step in the horizon."""
    values = series.to_numpy()
    n = len(values)
    horizon = config.FORECAST_HORIZON_WEEKS
    min_train = config.FORECAST_BACKTEST_MIN_TRAIN_WEEKS
    step = config.FORECAST_BACKTEST_STEP_WEEKS

    errors: dict[str, list[float]] = {name: [] for name in FORECASTERS}

    origin = min_train
    while origin + horizon <= n:
        history = values[:origin]
        actual = values[origin : origin + horizon]
        for name, fn in FORECASTERS.items():
            pred = fn(history, horizon)
            errors[name].extend(np.abs(actual - pred).tolist())
        origin += step

    return errors


def main() -> None:
    series = load_national_series()
    print(f"[forecasting] loaded {len(series)} real weekly observations for state={config.FORECAST_STATE}")

    errors = walk_forward_backtest(series)
    metrics = {}
    for name, err_list in errors.items():
        err_arr = np.array(err_list)
        mae_ci = bootstrap_ci(err_arr, np.mean, n_resamples=2000)
        metrics[name] = {"mae": float(err_arr.mean()), "mae_ci": mae_ci, "n_errors_pooled": len(err_arr)}
        print(f"[forecasting] {name}: MAE={err_arr.mean():.1f}  95% CI=({mae_ci['ci_low']:.1f}, {mae_ci['ci_high']:.1f})")

    best_name = min(metrics, key=lambda n: metrics[n]["mae"])
    naive_arr = np.array(errors["naive"])
    best_arr = np.array(errors[best_name])
    if best_name != "naive":
        sig = paired_bootstrap_test(naive_arr, best_arr, np.mean, n_resamples=2000)
        metrics["best_vs_naive_significance"] = sig
        print(f"[forecasting] {best_name} beats naive by {sig['observed_diff']:.1f} MAE, p={sig['p_value']:.4f}")

    metrics["selected_model"] = best_name

    final_forecast = FORECASTERS[best_name](series.to_numpy(), config.FORECAST_HORIZON_WEEKS)
    future_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(weeks=1), periods=config.FORECAST_HORIZON_WEEKS, freq="W-SAT"
    )
    forecast_df = pd.DataFrame({"week_ending_date": future_dates, "forecast_admissions": final_forecast})
    forecast_df.to_csv(config.RESULTS_DIR / "admissions_forecast.csv", index=False)

    with open(config.RESULTS_DIR / "forecasting_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"[forecasting] saved {best_name} forecast for next {config.FORECAST_HORIZON_WEEKS} weeks and metrics.")


if __name__ == "__main__":
    main()
