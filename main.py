from src.data_preparation import prepare_data
from src.train_model import train_models
from src.forecasting import generate_forecast
from src.visualization import create_all_visualizations
from src.simulation import run_simulation
from src.simulation_visualization import generate_simulation_graphs


def main():
    print("\nSTEP 1: DATA PREPARATION")
    prepare_data()
    print("\nSTEP 2: MODEL TRAINING")
    train_models()
    print("\nSTEP 3: FUTURE FORECASTING")
    generate_forecast(horizon=24)
    print("\nSTEP 4: MODEL VISUALIZATION")
    create_all_visualizations()
    print("\nSTEP 5: VIRTUAL SIMULATION")
    run_simulation(
        tariff_per_kwh=8.0,
        reduction_percent=5.0,
        peak_threshold_kw=3.0,
        daily_budget=500.0,
    )
    print("\nSTEP 6: SIMULATION GRAPHS")
    generate_simulation_graphs()
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
