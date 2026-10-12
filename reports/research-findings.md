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

