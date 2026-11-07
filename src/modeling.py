from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import RAW_FEATURES, TARGET, engineer_features

def build_models():
    def linear():
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", Ridge(alpha=2.0))])
    def tree(model):
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", model)])
    return {
        "Dummy baseline": tree(DummyRegressor(strategy="mean")),
        "Ridge regression": linear(),
        "Random forest": tree(RandomForestRegressor(
            n_estimators=220, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1)),
        "Extra trees": tree(ExtraTreesRegressor(
            n_estimators=240, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1)),
        "Histogram gradient boosting": tree(HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.08, l2_regularization=1.0,
            random_state=42))
    }

def metrics(y_true, pred):
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": float(r2_score(y_true, pred))
    }

def train_and_save(csv_path="data/facility_energy.csv", artifact_dir="artifacts"):
    data = pd.read_csv(csv_path)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", TARGET]).sort_values("timestamp")
    if len(data) < 300:
        raise ValueError("Need at least 300 valid hourly records to train reliably.")

    X_all = engineer_features(data[RAW_FEATURES])
    y_all = data[TARGET].astype(float)

    # Chronological 70/15/15 split: later periods are never used to train earlier ones.
    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X_train, y_train = X_all.iloc[:train_end], y_all.iloc[:train_end]
    X_val, y_val = X_all.iloc[train_end:val_end], y_all.iloc[train_end:val_end]
    X_test, y_test = X_all.iloc[val_end:], y_all.iloc[val_end:]

    results, fitted = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        row = {"Model": name, **metrics(y_val, val_pred)}
        results.append(row)
        fitted[name] = model

    comparison = pd.DataFrame(results).sort_values("MAE")
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted[best_name]
    test_metrics = metrics(y_test, best_model.predict(X_test))

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out / "best_model.joblib")
    comparison.to_csv(out / "model_comparison.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "test_metrics": test_metrics,
                   "train_rows": len(X_train), "validation_rows": len(X_val),
                   "test_rows": len(X_test), "data_note": "synthetic unless replaced"},
                  f, indent=2)

    # Permutation importance is computed on the held-out test period.
    pi = permutation_importance(best_model, X_test, y_test, n_repeats=6,
                                random_state=42, scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": pi.importances_mean
    }).sort_values("importance", ascending=False)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    return best_name, comparison, test_metrics
