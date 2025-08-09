# services/predictor.py
"""
Predictor service for the Smart Load Shedding Optimizer.

Responsibilities:
- Load per-district demand models (joblib files in models/saved)
- Load outage model (models/saved/outage_model.joblib) if present
- Provide helper to get latest snapshot from modeling dataset
- Predict demand for a snapshot and compute outage risk scores

Usage:
    from services.predictor import load_models, load_outage_model, get_latest_snapshot, predict_demand_on_snapshot, predict_outage_risk

Notes:
- This implementation predicts on an existing "snapshot" (one row per district, e.g. latest row).
  If you want multi-step forecasting, use a wrapper that generates future timestamps and rolling features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import glob
from typing import Dict, Tuple

_MODELS_CACHE: Dict[str, object] = {}
_OUTAGE_MODEL_CACHE = None

def load_demand_models(models_dir: str = "models/saved") -> Dict[str, object]:
    """
    Load demand models from models_dir. Expects filenames like 'demand_model_AP-01.joblib'.
    Returns dict: district_id -> model
    """
    global _MODELS_CACHE
    if _MODELS_CACHE:
        return _MODELS_CACHE

    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"[predictor] models directory not found: {models_dir}")
        _MODELS_CACHE = {}
        return _MODELS_CACHE

    models = {}
    for p in models_path.glob("demand_model_*.joblib"):
        try:
            m = joblib.load(p)
            # extract district id from filename: demand_model_AP-01.joblib -> AP-01
            name = p.stem  # e.g., demand_model_AP-01
            parts = name.split("_", 2)
            if len(parts) >= 3:
                district_id = parts[2]
            else:
                # fallback: use full stem
                district_id = parts[-1]
            models[district_id] = m
        except Exception as e:
            print(f"[predictor] failed loading model {p}: {e}")
    _MODELS_CACHE = models
    print(f"[predictor] loaded {len(models)} demand models")
    return models


def load_outage_model(path: str = "models/saved/outage_model.joblib"):
    """
    Load outage model info saved by outage_predictor (a dict with keys: 'model','feature_cols','threshold').
    """
    global _OUTAGE_MODEL_CACHE
    p = Path(path)
    if _OUTAGE_MODEL_CACHE is not None:
        return _OUTAGE_MODEL_CACHE
    if not p.exists():
        print(f"[predictor] outage model not found at {p}")
        _OUTAGE_MODEL_CACHE = None
        return None
    try:
        mi = joblib.load(p)
        _OUTAGE_MODEL_CACHE = mi
        print("[predictor] loaded outage model")
        return mi
    except Exception as e:
        print(f"[predictor] failed to load outage model: {e}")
        _OUTAGE_MODEL_CACHE = None
        return None


def get_latest_snapshot(modeling_path: str = "data/processed/modeling_dataset.parquet") -> pd.DataFrame:
    """
    Load modeling dataset and return latest row per district (one snapshot per district).
    """
    p = Path(modeling_path)
    if not p.exists():
        raise FileNotFoundError(modeling_path)
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["district_id", "timestamp"])
    latest = df.groupby("district_id").tail(1).reset_index(drop=True)
    return latest.copy()  # explicit copy to avoid SettingWithCopyWarning


def predict_demand_on_snapshot(models: Dict[str, object], snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict demand for each district in snapshot_df using per-district models.
    Returns DataFrame with columns: district_id, timestamp, peak_demand_mw_pred (or predicted_demand_mw)
    - models: dict district_id->model
    - snapshot_df: DataFrame with one row per district (must include district_id)
    """
    results = []
    snapshot = snapshot_df.copy()
    if "district_id" not in snapshot.columns:
        raise KeyError("snapshot_df must contain 'district_id'")

    for _, row in snapshot.iterrows():
        district = row["district_id"]
        ts = row.get("timestamp", pd.NaT)
        model = models.get(district)
        if model is None:
            # no model: fallback to using existing peak_demand_mw value if present
            fallback = None
            if "peak_demand_mw" in row.index:
                fallback = row["peak_demand_mw"]
            results.append({"district_id": district, "timestamp": ts, "predicted_peak_demand_mw": fallback})
            continue

        # Determine feature names expected by model
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            # try common attribute name for sklearn wrappers
            feature_names = getattr(model, "feature_names", None)

        if feature_names is None:
            # cannot determine features; attempt to pass all columns except identifiers
            X = pd.DataFrame([row.drop(labels=["district_id", "district_name"], errors="ignore")])
        else:
            # build X with required columns; fill missing with 0
            data = {}
            for f in feature_names:
                if f in row.index:
                    data[f] = row[f]
                else:
                    data[f] = 0
            X = pd.DataFrame([data])

        # ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            print(f"[predictor] model predict failed for {district}: {e}")
            pred = None
        results.append({"district_id": district, "timestamp": ts, "predicted_peak_demand_mw": float(pred) if pred is not None else None})

    return pd.DataFrame(results)


def predict_outage_risk(model_info, snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict outage risk (probability) for each district using the outage model_info.
    model_info: dict with keys 'model' and 'feature_cols'
    snapshot_df: DataFrame with one row per district (must include district_id)
    Returns DataFrame: district_id, timestamp, outage_risk
    """
    if model_info is None:
        # fallback: zero risk
        return pd.DataFrame([{"district_id": r, "timestamp": pd.NaT, "outage_risk": 0.0} for r in snapshot_df["district_id"].tolist()])

    model = model_info.get("model")
    feat_cols = model_info.get("feature_cols", [])

    snap = snapshot_df.copy()
    # prepare X
    X = pd.DataFrame()
    for f in feat_cols:
        if f in snap.columns:
            X[f] = pd.to_numeric(snap[f], errors="coerce").fillna(0)
        else:
            X[f] = 0.0
    # predict_proba
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"[predictor] outage model predict_proba failed: {e}")
        probs = np.zeros(len(X))
    out = pd.DataFrame({
        "district_id": snap["district_id"].values,
        "timestamp": snap.get("timestamp", pd.NaT).values,
        "outage_risk": probs
    })
    return out


# Convenience high-level function that runs full snapshot prediction
def predict_snapshot_all(model_dir: str = "models/saved", modeling_path: str = "data/processed/modeling_dataset.parquet") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load models, get latest snapshot, predict demand and outage risk.
    Returns (forecasts_df, risks_df)
    """
    models = load_demand_models(model_dir)
    outage_info = load_outage_model(Path(model_dir) / "outage_model.joblib")
    snap = get_latest_snapshot(modeling_path)
    forecasts = predict_demand_on_snapshot(models, snap)
    risks = predict_outage_risk(outage_info, snap)
    # merge timestamp alignment if needed
    if "timestamp" in snap.columns:
        forecasts["timestamp"] = snap.set_index("district_id").loc[forecasts["district_id"]]["timestamp"].values
        risks["timestamp"] = snap.set_index("district_id").loc[risks["district_id"]]["timestamp"].values
    return forecasts, risks


if __name__ == "__main__":
    # Quick smoke test when executed directly
    try:
        models = load_demand_models()
        outage_info = load_outage_model()
        snap = get_latest_snapshot()
        print(f"[predictor] snapshot has {len(snap)} districts")
        f, r = predict_snapshot_all()
        print("[predictor] forecasts sample:")
        print(f.head())
        print("[predictor] risks sample:")
        print(r.head())
    except Exception as e:
        print("predictor quicktest failed:", e)