# Methodology

1. Load the public UCI household electricity dataset.
2. Parse timestamps and numeric power readings.
3. Resample readings to hourly means, retaining missing timestamps.
4. Interpolate only short internal gaps and preserve longer gaps as missing.
5. Create calendar, lag, and rolling-average features without using the current target in rolling features.
6. Split valid feature rows chronologically into training and test periods.
7. Compare seasonal-naive, Linear Regression, and Random Forest predictions using MAE, RMSE, and R².
8. Select the ML candidate with the lowest test RMSE.
9. Generate recursive forecasts and a hypothetical cost/peak-threshold scenario.

## Limitations
The data is household-level, not commercial or industrial. Recursive errors can accumulate. Cost scenarios rely on user-supplied assumptions and are not measured savings. Use a separate validation or final test period for unbiased model assessment.
