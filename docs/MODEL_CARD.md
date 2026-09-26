# Forecast model card

## Intended use

Demonstrate short-horizon demand forecasting and benchmark discipline for staffing, replenishment, and capacity-planning workflows.

## Validation design

Rows are sorted by time and divided into the earliest 70% for training, the next 15% for validation, and the latest 15% for testing. No random split is used. Lag and rolling features are shifted so the current target never enters its own predictors.

## Baseline

The seasonal naive forecast uses demand from the same hour one week earlier for hourly data, or the same weekday one week earlier for daily data. The gradient-boosting model must outperform this baseline on validation mean absolute error to be selected.

## Leakage controls

The `casual` and `registered` fields are removed because their sum equals total demand. Features that would not exist at prediction time must also be excluded in any production adaptation.

## Limitations

- two historical years from one bike-sharing system;
- observed rentals can be constrained by bike availability;
- weather forecasts are not the same as observed weather;
- no promotions, prices, product hierarchy, or inventory;
- no prediction intervals in the baseline implementation.

## Production requirements

A retail implementation would require rolling-origin backtesting, prediction intervals, promotion and price features, stock-out flags, product-location hierarchy, forecast reconciliation, drift monitoring, and an agreed operational loss function.
