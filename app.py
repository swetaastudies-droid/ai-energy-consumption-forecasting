from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Find the project folder automatically
BASE_DIR = Path(__file__).resolve().parent

FORECAST_FILE = BASE_DIR / "outputs" / "forecasts" / "future_forecast.csv"
METRICS_FILE = BASE_DIR / "outputs" / "metrics" / "model_comparison.csv"
SIMULATION_FILE = BASE_DIR / "outputs" / "simulation" / "scenario_hourly.csv"
HISTORY_FILE = BASE_DIR / "data" / "processed" / "hourly_energy.csv"

st.title("⚡ AI-Powered Energy Consumption Forecasting")
st.caption(
    "Historical analysis | Machine learning forecasts | "
    "Virtual energy-cost simulation"
)

# --------------------------------------------------
# 2. Load available project results
# --------------------------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# --------------------------------------------------
# 3. Model evaluation
# --------------------------------------------------
st.header("1. Model Evaluation")

if METRICS_FILE.exists():
    metrics = load_csv(METRICS_FILE)
    st.dataframe(metrics, use_container_width=True)

    # Display the best model if the necessary columns exist
    if {"model", "RMSE_kW"}.issubset(metrics.columns):
        best_row = metrics.loc[metrics["RMSE_kW"].idxmin()]
        st.metric("Lowest RMSE model", str(best_row["model"]))
else:
    st.warning(
        "Model metrics not found. Run the model-training script first."
    )

# --------------------------------------------------
# # 4. Historical Energy Consumption
st.header("4. Historical Energy Consumption")

if HISTORY_FILE.exists():
    history = pd.read_csv(HISTORY_FILE)
    history.columns = history.columns.str.strip()

    if {"timestamp", "energy_consumption_kw"}.issubset(history.columns):
        history["timestamp"] = pd.to_datetime(
            history["timestamp"], errors="coerce"
        )
        history["energy_consumption_kw"] = pd.to_numeric(
            history["energy_consumption_kw"], errors="coerce"
        )

        history = history.dropna(
            subset=["timestamp", "energy_consumption_kw"]
        ).sort_values("timestamp")

        if not history.empty:
            daily = (
                history.set_index("timestamp")["energy_consumption_kw"]
                .resample("D")
                .mean()
                .dropna()
                .reset_index()
            )

            fig = px.line(
                daily,
                x="timestamp",
                y="energy_consumption_kw",
                title="Average Daily Energy Consumption",
                labels={
                    "timestamp": "Date",
                    "energy_consumption_kw": "Average Power (kW)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(daily.tail(10), use_container_width=True)
        else:
            st.warning("The historical CSV has no valid rows to display.")
    else:
        st.warning(
            f"Unexpected CSV columns: {history.columns.tolist()}"
        )
else:
    st.warning(
        "Historical data not found. Run: python -m src.data_preparation"
    )

# --------------------------------------------------
# 5. Future forecasts
# --------------------------------------------------
st.header("3. Future Energy Forecast")

if FORECAST_FILE.exists():
    forecast = load_csv(FORECAST_FILE)

    if "timestamp" in forecast.columns:
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

    if "predicted_consumption_kw" in forecast.columns:
        horizon = st.selectbox(
            "Choose forecast horizon",
            [24, 48, 72, 168],
            index=0
        )

        view = forecast.head(horizon)

        fig = px.line(
            view,
            x="timestamp",
            y="predicted_consumption_kw",
            markers=True,
            title=f"Predicted Consumption — Next {horizon} Hours",
            labels={
                "timestamp": "Forecast time",
                "predicted_consumption_kw": "Predicted power (kW)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True)

        st.download_button(
            "Download Forecast CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="energy_forecast.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "The forecast file does not contain the expected prediction column."
        )
else:
    st.warning(
        "Forecast file not found. Run: python -m src.forecasting"
    )

# --------------------------------------------------
# 6. Virtual simulation
# --------------------------------------------------
st.header("4. Virtual Energy-Cost Simulation")

st.caption(
    "The following scenario uses assumptions for planning. "
    "It does not represent verified real-world savings."
)

if SIMULATION_FILE.exists():
    scenario = load_csv(SIMULATION_FILE)

    st.dataframe(scenario.head(24), use_container_width=True)

    cost_columns = [
        col for col in scenario.columns
        if "baseline_cost" in col.lower()
        or "scenario_cost" in col.lower()
    ]

    if len(cost_columns) >= 2:
        fig = px.line(
            scenario,
            y=cost_columns[:2],
            title="Baseline Cost vs Scenario Cost",
            labels={
                "value": "Estimated cost (₹)",
                "index": "Forecast hour",
                "variable": "Scenario"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Simulation CSV",
        data=scenario.to_csv(index=False).encode("utf-8"),
        file_name="energy_simulation.csv",
        mime="text/csv"
    )
else:
    st.warning(
        "Simulation file not found. Run: python -m src.simulation"
    )

st.divider()
st.caption(
    "Student portfolio project using historical household energy data "
    "and hypothetical simulation scenarios."
)