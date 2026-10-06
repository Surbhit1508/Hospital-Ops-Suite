import numpy as np

from src.forecasting.train_admissions_forecast import naive_forecast, seasonal_naive_forecast


def test_naive_forecast_repeats_last_value():
    history = np.array([1.0, 2.0, 3.0, 10.0])
    forecast = naive_forecast(history, horizon=5)
    assert len(forecast) == 5
    assert (forecast == 10.0).all()


def test_seasonal_naive_repeats_last_season_when_enough_history():
    history = np.arange(1, 105, dtype=float)  # 104 points, season=52
    forecast = seasonal_naive_forecast(history, horizon=4, season=52)
    expected = history[-52 : -52 + 4]
    assert np.allclose(forecast, expected)


def test_seasonal_naive_falls_back_to_naive_when_history_too_short():
    history = np.array([5.0, 6.0, 7.0])
    forecast = seasonal_naive_forecast(history, horizon=3, season=52)
    assert (forecast == 7.0).all()
