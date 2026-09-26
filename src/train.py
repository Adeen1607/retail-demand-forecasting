"""Train and evaluate a time-aware demand forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ucimlrepo import fetch_ucirepo


TARGET = "demand"


def load_data() -> tuple[pd.DataFrame, int]:
    dataset = fetch_ucirepo(id=275)
    features = dataset.data.features.copy()
    target = dataset.data.targets.squeeze().rename(TARGET)

    data = pd.concat([features, target], axis=1)
    data = data.drop(columns=["casual", "registered"], errors="ignore")

    if "dteday" in data.columns:
        date = pd.to_datetime(data["dteday"], errors="coerce")
        hourly = "hr" in data.columns
        hour = pd.to_numeric(data["hr"], errors="coerce").fillna(0) if hourly else 0
        data["timestamp"] = date + pd.to_timedelta(hour, unit="h")
    else:
        hourly = "hr" in data.columns
        data["timestamp"] = pd.RangeIndex(len(data))

    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    seasonal_lag = 168 if hourly else 7
    return data, seasonal_lag


def add_history_features(data: pd.DataFrame, seasonal_lag: int) -> pd.DataFrame:
    frame = data.copy()
    lags = sorted({1, 24 if seasonal_lag == 168 else 1, seasonal_lag})
    for lag in lags:
        frame[f"lag_{lag}"] = frame[TARGET].shift(lag)

    windows = [24, 168] if seasonal_lag == 168 else [7, 28]
    for window in windows:
        history = frame[TARGET].shift(1).rolling(window)
        frame[f"rolling_mean_{window}"] = history.mean()
        frame[f"rolling_std_{window}"] = history.std()

    if pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        frame["hour_of_day"] = frame["timestamp"].dt.hour
        frame["day_of_week"] = frame["timestamp"].dt.dayofweek
        frame["month_number"] = frame["timestamp"].dt.month

    return frame.dropna().reset_index(drop=True)


def split_chronologically(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    return (
        data.iloc[:train_end].copy(),
        data.iloc[train_end:validation_end].copy(),
        data.iloc[validation_end:].copy(),
    )


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_array = actual.to_numpy(dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = (np.abs(actual_array) + np.abs(predicted_array)) / 2
    smape = np.mean(
        np.divide(
            np.abs(actual_array - predicted_array),
            denominator,
            out=np.zeros_like(actual_array),
            where=denominator != 0,
        )
    )
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "smape": float(smape),
    }


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET, "timestamp", "dteday", "instant"}
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]


def save_plot(results: pd.DataFrame, output_dir: Path) -> None:
    tail = results.tail(24 * 14 if len(results) >= 24 * 14 else len(results))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(tail["timestamp"], tail["actual"], label="Actual", color="#0F172A")
    axis.plot(
        tail["timestamp"],
        tail["selected_forecast"],
        label="Forecast",
        color="#2563EB",
    )
    axis.set(title="Demand Forecast — Latest Test Period", ylabel="Demand")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "forecast_vs_actual.png", dpi=160)
    plt.close(figure)


def train(output_dir: Path, seed: int) -> None:
    data, seasonal_lag = load_data()
    engineered = add_history_features(data, seasonal_lag)
    train_set, validation_set, test_set = split_chronologically(engineered)
    columns = feature_columns(engineered)

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train_set[columns], train_set[TARGET])

    validation_model = model.predict(validation_set[columns])
    validation_naive = validation_set[f"lag_{seasonal_lag}"].to_numpy()
    validation_metrics = {
        "gradient_boosting": metrics(validation_set[TARGET], validation_model),
        "seasonal_naive": metrics(validation_set[TARGET], validation_naive),
    }
    selected_model = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["mae"],
    )

    test_model = model.predict(test_set[columns])
    test_naive = test_set[f"lag_{seasonal_lag}"].to_numpy()
    selected_forecast = test_model if selected_model == "gradient_boosting" else test_naive

    report = {
        "selected_model": selected_model,
        "seasonal_lag": seasonal_lag,
        "validation": validation_metrics,
        "test_gradient_boosting": metrics(test_set[TARGET], test_model),
        "test_seasonal_naive": metrics(test_set[TARGET], test_naive),
        "test_selected": metrics(test_set[TARGET], selected_forecast),
        "train_rows": len(train_set),
        "validation_rows": len(validation_set),
        "test_rows": len(test_set),
        "random_seed": seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    results = pd.DataFrame(
        {
            "timestamp": test_set["timestamp"],
            "actual": test_set[TARGET],
            "gradient_boosting_forecast": test_model,
            "seasonal_naive_forecast": test_naive,
            "selected_forecast": selected_forecast,
        }
    )
    results.to_csv(output_dir / "test_forecasts.csv", index=False)
    joblib.dump(
        {"model": model, "feature_columns": columns, "seasonal_lag": seasonal_lag},
        output_dir / "demand_forecast.joblib",
    )
    save_plot(results, output_dir)
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train demand forecast.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.output_dir, arguments.seed)
