import pandas as pd

RAW_FEATURES = [
    "timestamp", "temperature_c", "humidity_pct", "occupancy_pct",
    "active_buildings", "operating_hours", "is_holiday",
    "previous_hour_kwh", "energy_price_per_kwh"
]
TARGET = "energy_kwh"

NUMERIC_FEATURES = [
    "temperature_c", "humidity_pct", "occupancy_pct", "active_buildings",
    "operating_hours", "is_holiday", "previous_hour_kwh",
    "energy_price_per_kwh", "hour", "day_of_week", "month",
    "is_weekend", "is_business_hours", "cooling_degree",
    "heating_degree", "occupancy_building_load"
]

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable calendar and operational features from raw input columns."""
    df = frame.copy()
    if "timestamp" not in df:
        raise ValueError("Missing required column: timestamp")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Some timestamp values are invalid.")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_business_hours"] = ts.dt.hour.between(7, 19).astype(int)
    df["cooling_degree"] = (df["temperature_c"] - 22).clip(lower=0)
    df["heating_degree"] = (18 - df["temperature_c"]).clip(lower=0)
    df["occupancy_building_load"] = (
        df["occupancy_pct"] * df["active_buildings"] / 100.0
    )
    return df[NUMERIC_FEATURES]
