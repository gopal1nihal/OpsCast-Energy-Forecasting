# OpsCast — Facility Energy Demand & Cost Forecasting

An end-to-end regression project for estimating hourly electricity consumption and operational energy cost from weather, occupancy, calendar, and facility-operation signals.

## What it does
- Creates a reproducible synthetic training dataset (clearly labelled as synthetic).
- Performs data validation, missing-value imputation, and feature engineering.
- Compares a baseline, Ridge, Random Forest, Extra Trees, and HistGradientBoosting regressors.
- Selects a model using validation MAE and evaluates it on a held-out test set.
- Saves the trained pipeline and metrics.
- Provides a Streamlit dashboard for predictions, historical analysis, and feature importance.

> **Honesty note:** The included data is synthetic so the project runs immediately. Do not describe it as real campus/company data. For a production deployment, replace it with permissioned meter readings and matching occupancy/weather data, retrain, and monitor model drift.

## Project structure
```text
opscast/
├── app.py
├── train.py
├── generate_data.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── data/
│   └── README.md
└── src/
    ├── __init__.py
    ├── features.py
    └── modeling.py
```

## 1. Run locally in VS Code

Install Python 3.10–3.12, then open this folder in VS Code.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 2. Input data format

Replace `data/facility_energy.csv` with your real, authorised data using these columns:

| Column | Meaning |
|---|---|
| timestamp | Local date/time in ISO format |
| temperature_c | Outdoor temperature in °C |
| humidity_pct | Relative humidity, 0–100 |
| occupancy_pct | Estimated occupancy, 0–100 |
| active_buildings | Number of buildings operating |
| operating_hours | Operating hours for the day |
| is_holiday | 1 if holiday, otherwise 0 |
| previous_hour_kwh | Previous hour's meter reading / consumption |
| energy_price_per_kwh | Tariff in your currency per kWh |
| energy_kwh | Actual hourly energy consumption (target) |

The sample generator creates 18 months of hourly synthetic records with realistic patterns and a small amount of missing input data.

## 3. Train and evaluate
```bash
python generate_data.py
python train.py
```
Training saves:
- `artifacts/best_model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_comparison.csv`
- `artifacts/permutation_importance.csv`

Model selection uses validation MAE. The test set is kept separate until final evaluation. The time-based split avoids randomly mixing future observations into training.

## 4. Deploy

### Streamlit Community Cloud
1. Push the project to a public GitHub repository.
2. Ensure `requirements.txt`, `app.py`, `train.py`, `generate_data.py`, `src/`, and `data/` are committed.
3. Generate data and train locally first.
4. Commit the `artifacts/` folder (the trained model is required by the hosted app).
5. In Streamlit Community Cloud, create an app from your repository and select `app.py` as the entry point.

For a fresh deployment where artifacts are absent, run the training commands locally and push the resulting artifacts. Do not commit private meter data or secrets.

### Docker
```bash
docker build -t opscast .
docker run -p 8501:8501 opscast
```

## 5. GitHub
```bash
git init
git add .
git commit -m "Build OpsCast energy forecasting platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/opscast.git
git push -u origin main
```

## 6. Before calling it production-ready
- Validate meter units, time zones, missing intervals, and tariff schedules.
- Use real historical meter readings and authorised occupancy data.
- Retrain on a rolling schedule and monitor MAE/RMSE by season and building.
- Compare against a seasonal-naive baseline.
- Add authentication, access control, audit logs, and secure data storage before handling sensitive operational data.
- Treat forecasts as decision support, not guaranteed savings.

## Resume bullet (after you have run and deployed it)
**OpsCast — Facility Energy Demand & Cost Forecasting | Python, Pandas, NumPy, Scikit-learn, Streamlit**
- Developed an end-to-end regression pipeline to forecast hourly facility energy consumption using weather, occupancy, calendar, and operational features.
- Built time-aware preprocessing and model-selection workflows comparing linear and tree-based regressors using MAE, RMSE, and R².
- Deployed an interactive dashboard for consumption and cost estimates, historical trends, and model explainability.

Only include measured performance numbers after training on a credible real dataset. Clearly label synthetic-data results.
