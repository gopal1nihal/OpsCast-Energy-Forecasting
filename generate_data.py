from pathlib import Path
import numpy as np
import pandas as pd

def generate_dataset(path="data/facility_energy.csv", seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=18 * 30 * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    temperature = 27 + 6*np.sin(2*np.pi*(day_of_year-80)/365) + 3*np.sin(2*np.pi*(hour-14)/24)
    temperature += rng.normal(0, 1.8, len(timestamps))
    humidity = np.clip(66 - 0.7*(temperature-27) + rng.normal(0, 7, len(timestamps)), 25, 98)
    weekend = (dow >= 5)
    holiday = ((timestamps.month == 1) & (timestamps.day == 1)) | ((timestamps.month == 8) & (timestamps.day == 15))
    business = (hour >= 7) & (hour <= 19) & (~weekend) & (~holiday)
    occupancy = np.where(business, 48 + 30*np.sin(np.pi*(hour-7)/12), 7)
    occupancy = np.clip(occupancy + rng.normal(0, 8, len(timestamps)), 0, 100)
    buildings = np.where(business, rng.integers(5, 9, len(timestamps)), rng.integers(1, 5, len(timestamps)))
    operating_hours = np.where(business, 12, 3)
    tariff = np.where((hour >= 18) & (hour <= 22), 9.2, 6.4)
    tariff = tariff + rng.normal(0, 0.15, len(timestamps))

    # Simulated load has base, occupancy, weather/HVAC, calendar and noise components.
    cooling = np.maximum(temperature - 22, 0)
    heating = np.maximum(18 - temperature, 0)
    energy = (24 + 0.55*occupancy + 3.8*buildings + 1.9*cooling +
              0.7*heating + 0.08*humidity + 0.9*business +
              4.5*np.sin(2*np.pi*(hour-8)/24)**2 +
              rng.normal(0, 3.2, len(timestamps)))
    energy = np.maximum(energy, 5)
    previous = np.roll(energy, 1)
    previous[0] = energy[0]
    frame = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "temperature_c": temperature.round(2),
        "humidity_pct": humidity.round(2),
        "occupancy_pct": occupancy.round(2),
        "active_buildings": buildings,
        "operating_hours": operating_hours,
        "is_holiday": holiday.astype(int),
        "previous_hour_kwh": previous.round(2),
        "energy_price_per_kwh": tariff.round(2),
        "energy_kwh": energy.round(2)
    })
    # Introduce a small amount of missing feature values to exercise imputation.
    for col in ["humidity_pct", "occupancy_pct", "previous_hour_kwh"]:
        idx = rng.choice(frame.index, size=int(len(frame)*0.012), replace=False)
        frame.loc[idx, col] = np.nan

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Generated {len(frame):,} synthetic hourly records: {path}")

if __name__ == "__main__":
    generate_dataset()
