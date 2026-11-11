import numpy as np
import pandas as pd
import pytest

from src.train import TARGET, add_history_features, metrics, split_chronologically


def test_history_features_use_only_prior_targets() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D"),
            TARGET: np.arange(1.0, 41.0),
        }
    )

    engineered = add_history_features(data, seasonal_lag=7)

    first = engineered.iloc[0]
    assert first["timestamp"] == pd.Timestamp("2026-01-29")
    assert first["lag_1"] == 28.0
    assert first["lag_7"] == 22.0
    assert first["rolling_mean_7"] == pytest.approx(np.mean(np.arange(22.0, 29.0)))
    assert first["rolling_mean_28"] == pytest.approx(np.mean(np.arange(1.0, 29.0)))


def test_chronological_split_preserves_order_and_sizes() -> None:
    frame = pd.DataFrame({"sequence": range(20)})

    train, validation, test = split_chronologically(frame)

    assert len(train) == 14
    assert len(validation) == 3
    assert len(test) == 3
    assert train["sequence"].max() < validation["sequence"].min()
    assert validation["sequence"].max() < test["sequence"].min()


def test_metrics_are_zero_for_a_perfect_forecast() -> None:
    actual = pd.Series([0.0, 10.0, 25.0])
    predicted = np.array([0.0, 10.0, 25.0])

    result = metrics(actual, predicted)

    assert result == {"mae": 0.0, "rmse": 0.0, "smape": 0.0}


@pytest.mark.parametrize(
    "dates,hours,error",
    [
        (["2026-01-01", "invalid"], [0, 1], "invalid dates"),
        (["2026-01-01", "2026-01-01"], [0, 24], "invalid hour"),
        (["2026-01-01", "2026-01-01"], [0, -1], "invalid hour"),
        (["2026-01-01", "2026-01-01"], [0, 1.5], "invalid hour"),
    ],
)
def test_load_data_rejects_invalid_calendar_inputs(monkeypatch, dates, hours, error):
    from types import SimpleNamespace
    from src import train as training

    features = pd.DataFrame({"dteday": dates, "hr": hours})
    dataset = SimpleNamespace(data=SimpleNamespace(
        features=features, targets=pd.DataFrame({"cnt": [1, 2]})
    ))
    monkeypatch.setattr(training, "fetch_ucirepo", lambda id: dataset)
    with pytest.raises(ValueError, match=error):
        training.load_data()
