from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "hourly_energy.csv"


def prepare_data():
    """Load the UCI dataset and create hourly average power readings."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Download and extract the UCI dataset into data/raw/."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(
        RAW_FILE,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"],
        low_memory=False,
    )
    data["timestamp"] = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    data["Global_active_power"] = pd.to_numeric(
        data["Global_active_power"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"])
    data = data.drop_duplicates(subset=["timestamp"])
    data = data.sort_values("timestamp")
    data = data[["timestamp", "Global_active_power"]].copy()
    data.loc[data["Global_active_power"] < 0, "Global_active_power"] = float("nan")
    data = data.set_index("timestamp")

    hourly = data["Global_active_power"].resample("h").agg(["mean", "count"])
    hourly.loc[hourly["count"] < 45, "mean"] = float("nan")
    hourly = hourly[["mean"]].rename(
        columns={"mean": "energy_consumption_kw"}
    )
    hourly["energy_consumption_kw"] = hourly[
        "energy_consumption_kw"
    ].interpolate(method="time", limit=2, limit_area="inside")

    # Keep missing hours in the timeline; downstream features will exclude them.
    hourly = hourly.reset_index()
    hourly.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed data saved to: {OUTPUT_FILE}")
    print(f"Hourly rows: {len(hourly):,}")
    print(f"Missing hourly values: {hourly['energy_consumption_kw'].isna().sum():,}")
    return hourly


if __name__ == "__main__":
    prepare_data()
