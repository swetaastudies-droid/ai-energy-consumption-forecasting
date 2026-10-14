# AI-Powered Energy Consumption Forecasting System

A beginner-friendly machine-learning project that uses public household electricity data to forecast hourly power consumption and explore hypothetical cost and peak-demand scenarios.

## Features
- Data cleaning and hourly resampling
- Time-based, lag, and rolling-average features
- Seasonal-naive, Linear Regression, and Random Forest comparison
- MAE, RMSE, and R² evaluation
- Recursive forecasts for 1–168 hours
- Virtual tariff, budget, and peak-threshold simulation
- Saved charts and CSV/JSON outputs

## Dataset
Download the **Individual Household Electric Power Consumption** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption). Extract `household_power_consumption.txt` into `data/raw/`.

The dataset contains household data; results should not be treated as validated industrial forecasts.

## Setup
Python 3.12 or 3.13 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run
Run the full workflow:
```bash
python main.py
```

Or run individual stages:
```bash
python -m src.data_preparation
python -m src.train_model
python -m src.forecasting
python -m src.visualization
python -m src.simulation
python -m src.simulation_visualization
```

## Main outputs
- `data/processed/hourly_energy.csv`
- `models/best_model.joblib`
- `outputs/metrics/model_comparison.csv`
- `outputs/forecasts/test_predictions.csv`
- `outputs/forecasts/future_forecast.csv`
- `outputs/simulation/scenario_hourly.csv`
- `outputs/simulation/scenario_summary.json`
- `outputs/figures/`

## Interpretation and limitations
Simulation tariff, budget, peak threshold, and reduction percentages are illustrative. Estimated differences are not verified savings. The final test period is used to select the best ML candidate, so an additional untouched evaluation period is recommended for independent assessment.

## Dataset attribution
Please retain attribution to the UCI Machine Learning Repository and review the dataset's license and citation requirements before redistributing the data.
