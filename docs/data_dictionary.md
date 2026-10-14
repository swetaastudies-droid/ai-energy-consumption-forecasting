# Data Dictionary

## Processed hourly data
- `timestamp`: hourly timestamp.
- `energy_consumption_kw`: hourly mean of `Global_active_power`, in kW. For a one-hour interval, this average power corresponds numerically to estimated kWh for that hour if the interval is adequately represented.

## Forecast output
- `timestamp`: future hourly timestamp.
- `predicted_consumption_kw`: model-predicted average power in kW.
- `forecast_step`: hour ahead (1 to requested horizon).
- `model`: selected model name.

## Simulation output
- `baseline_energy_kwh`: estimated baseline energy for each one-hour interval.
- `baseline_cost_inr`: baseline estimated cost.
- `scenario_energy_kwh`: energy after the hypothetical reduction.
- `scenario_cost_inr`: scenario estimated cost.
- `estimated_cost_difference_inr`: baseline cost minus scenario cost.
- `peak_threshold_kw`: configured threshold.
- `peak_alert`: whether predicted power exceeds the threshold.
