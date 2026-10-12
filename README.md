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


## Rebuild the reading edition

```bash
python -m pip install -r requirements-research.txt
python scripts/build_research_site.py
```

This renders saved notebook outputs without rerunning model training or external APIs. To refresh findings, run the analysis commands above first, update `reports/research-findings.md`, and rebuild.

<!-- research-edition:start -->
## Research edition

[Research report and final visual results](research/report.md) · [Browser-ready notebook catalogue](research/notebooks.html)

Fresh execution on 11 October 2026 using the committed pipeline and random seed 42. Validation selected **gradient_boosting**. On 2,582 later-period rows, selected-model MAE was **32.19** and seasonal-naive MAE was **85.74**, a **62.5%** reduction. This is rolling one-step evaluation using observed prior targets, not an advance multi-step forecast.

| Measure | Value |
|---|---:|
| selected model | gradient_boosting |
| seasonal lag | 168 |
| train rows | 12047 |
| validation rows | 2582 |
| test rows | 2582 |
| random seed | 42 |


| Test model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| test_gradient_boosting | 32.19 | 51.45 | 0.221 |
| test_seasonal_naive | 85.74 | 149.41 | 0.456 |
| test_selected | 32.19 | 51.45 | 0.221 |



The `research/` folder contains a professional static reading edition. The Pages workflow deploys it after GitHub Pages is enabled with **GitHub Actions** as the publishing source. Intended URL: `https://adeen1607.github.io/retail-demand-forecasting/`. Deployment status must be checked in Actions before treating that address as live.

<!-- research-edition:end -->
