# Retail Demand Forecasting

A time-series forecasting case study that predicts hourly bike-rental demand and demonstrates the same planning disciplines used for retail staffing, replenishment, and capacity decisions.

## Business objective

Produce next-period demand estimates that can support inventory, labour, and capacity planning while clearly benchmarking model performance against a seasonal naive forecast.

## Data source

The project uses the [Bike Sharing dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). It contains hourly and daily rental counts from the Capital Bikeshare system for 2011 and 2012, together with calendar and weather attributes.

## Forecasting workflow

1. Fetch the official dataset.
2. Sort observations chronologically.
3. Remove target-leaking fields whose sum equals total demand.
4. Create lag and rolling-history features using only prior observations.
5. Split data chronologically into train, validation, and test periods.
6. Compare a 168-hour seasonal naive baseline with gradient boosting.
7. Lock the selected model and evaluate on the latest test period.
8. Export forecasts, error metrics, and time-series diagnostics.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Metrics

- mean absolute error;
- root mean squared error;
- symmetric mean absolute percentage error;
- error by hour, weekday, and demand level;
- improvement over the seasonal naive baseline.

## Responsible use

This historical dataset is used to demonstrate forecasting design. A retail deployment would require current product-location demand, promotions, stock availability, price, holidays, product substitutions, and explicit treatment of censored demand caused by stock-outs.
