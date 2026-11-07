from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import RAW_FEATURES, engineer_features

st.set_page_config(page_title="OpsCast | Energy Intelligence", page_icon="⚡",
                   layout="wide")
ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "best_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_history():
    path = Path("data/facility_energy.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"])

st.title("⚡ OpsCast")
st.caption("Facility energy demand & operational cost forecasting")
st.info("The project includes sample energy consumption data for model training and testing. Real facility data can be integrated for practical use.")

model = load_model()
if model is None:
    st.error("Trained model not found. In the VS Code terminal, run: `python generate_data.py` followed by `python train.py`.")
    st.stop()

try:
    metrics_info = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
except Exception:
    metrics_info = {}

history = load_history()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected model", metrics_info.get("selected_model", "Trained model"))
test = metrics_info.get("test_metrics", {})
m2.metric("Test MAE", f'{test.get("MAE", float("nan")):.2f} kWh')
m3.metric("Test RMSE", f'{test.get("RMSE", float("nan")):.2f} kWh')
m4.metric("Test R²", f'{test.get("R2", float("nan")):.3f}')

st.divider()
left, right = st.columns([1, 1.25])
with left:
    st.subheader("Forecast a facility-hour")
    with st.form("forecast_form"):
        forecast_time = st.datetime_input("Forecast timestamp", value=pd.Timestamp.now().floor("h").to_pydatetime())
        temperature = st.slider("Outdoor temperature (°C)", -5.0, 48.0, 29.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 65)
        occupancy = st.slider("Expected occupancy (%)", 0, 100, 55)
        buildings = st.number_input("Active buildings", min_value=1, max_value=100, value=6)
        hours = st.slider("Operating hours today", 0, 24, 12)
        holiday = st.selectbox("Holiday?", ["No", "Yes"])
        previous = st.number_input("Previous-hour consumption (kWh)", min_value=0.0, value=75.0, step=1.0)
        tariff = st.number_input("Energy tariff per kWh", min_value=0.0, value=7.0, step=0.1)
        submitted = st.form_submit_button("Generate forecast", type="primary")

with right:
    st.subheader("Forecast result")
    if submitted:
        raw = pd.DataFrame([{
            "timestamp": pd.Timestamp(forecast_time),
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "occupancy_pct": occupancy,
            "active_buildings": buildings,
            "operating_hours": hours,
            "is_holiday": int(holiday == "Yes"),
            "previous_hour_kwh": previous,
            "energy_price_per_kwh": tariff
        }], columns=RAW_FEATURES)
        prediction = max(0.0, float(model.predict(engineer_features(raw))[0]))
        cost = prediction * tariff
        a, b = st.columns(2)
        a.metric("Predicted consumption", f"{prediction:,.2f} kWh")
        b.metric("Estimated energy cost", f"₹{cost:,.2f}")
        if prediction >= 100:
            st.warning("High-load estimate: review operating schedules and cooling demand.")
        else:
            st.success("Forecast generated. Compare it with your facility's operating plan.")
        st.caption("Cost is a simple estimate: predicted kWh × entered tariff. Taxes, demand charges and tariff slabs are not included.")
    else:
        st.write("Enter operating conditions and select **Generate forecast** to estimate consumption and cost.")
        st.caption("The model returns an estimate, not a guaranteed reading or savings commitment.")

st.divider()
if history is not None:
    st.subheader("Historical demand explorer")
    period = history.copy()
    period["date"] = period["timestamp"].dt.date
    daily = period.groupby("date", as_index=False)["energy_kwh"].sum()
    fig = px.line(daily.tail(45), x="date", y="energy_kwh",
                  title="Daily energy consumption — latest 45 days",
                  labels={"date": "Date", "energy_kwh": "Energy (kWh)"})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        hourly = period.groupby(period["timestamp"].dt.hour, as_index=False)["energy_kwh"].mean()
        hourly.columns = ["hour", "average_kwh"]
        st.plotly_chart(px.bar(hourly, x="hour", y="average_kwh",
                               title="Average consumption by hour"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(period.sample(min(1200, len(period)), random_state=42),
                                   x="occupancy_pct", y="energy_kwh",
                                   color="temperature_c",
                                   title="Occupancy vs consumption",
                                   labels={"occupancy_pct": "Occupancy (%)",
                                           "energy_kwh": "Energy (kWh)",
                                           "temperature_c": "Temperature (°C)"}),
                        use_container_width=True)

importance_path = ARTIFACTS / "permutation_importance.csv"
if importance_path.exists():
    st.subheader("What influences the model?")
    imp = pd.read_csv(importance_path).head(10).sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 title="Permutation importance (higher means more impact on validation signal)")
    st.plotly_chart(fig, use_container_width=True)

comparison_path = ARTIFACTS / "model_comparison.csv"
if comparison_path.exists():
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(comparison_path).round(3), use_container_width=True, hide_index=True)

st.caption("OpsCast • Regression-based ML • Python / Pandas / NumPy / Scikit-learn / Streamlit")
