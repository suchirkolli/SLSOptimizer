"""Prediction helpers for the Smart Load Shedding Optimizer."""

from __future__ import annotations
from services.features import add_time_features, add_lag_features
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd

_MODELS_CACHE: Dict[str, object] = {}
_OUTAGE_MODEL_CACHE = None
_HISTORY_CACHE = None
_DISTRICT_MAP_CACHE = None


def load_demand_models(models_dir: str = "models/saved") -> Dict[str, object]:
    global _MODELS_CACHE
    if _MODELS_CACHE:
        return _MODELS_CACHE

    models = {}
    new_model_path = Path(models_dir) / "demand_model_NEW.joblib"

    if new_model_path.exists():
        try:
            global_model = joblib.load(new_model_path)
            history = load_modeling_history()
            district_ids = sorted(history["district_id"].dropna().astype(str).unique())

            for district_id in district_ids:
                models[district_id] = global_model

            _MODELS_CACHE = models
            print(f"[predictor] loaded retrained global demand model for {len(models)} districts")
            return models
        except Exception as e:
            print(f"[predictor] failed loading retrained model {new_model_path}: {e}")

    for p in Path(models_dir).glob("demand_model_*.joblib"):
        if p.name == "demand_model_NEW.joblib":
            continue
        try:
            model = joblib.load(p)
            district_id = p.stem.split("_")[-1]
            models[district_id] = model
        except Exception as e:
            print(f"[predictor] failed loading model {p}: {e}")

    _MODELS_CACHE = models
    print(f"[predictor] loaded {len(models)} legacy demand models")
    return models


def load_outage_model(path: str | Path = "models/saved/outage_model.joblib"):
    global _OUTAGE_MODEL_CACHE
    if _OUTAGE_MODEL_CACHE is not None:
        return _OUTAGE_MODEL_CACHE
    p = Path(path)
    if not p.exists():
        print(f"[predictor] outage model not found at {p}")
        return None
    try:
        _OUTAGE_MODEL_CACHE = joblib.load(p)
        print("[predictor] loaded outage model")
        return _OUTAGE_MODEL_CACHE
    except Exception as e:
        print(f"[predictor] failed to load outage model: {e}")
        return None


def _read_parquet_robust(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_parquet(p)
    except ImportError as e:
        raise ImportError(
            f"Parquet support is missing for {p}. Install pyarrow with `pip install pyarrow`."
        ) from e


def _get_district_name_map(ref_path: str | Path = "data/geographic/ap_districts_reference.csv") -> dict:
    global _DISTRICT_MAP_CACHE
    if _DISTRICT_MAP_CACHE is not None:
        return _DISTRICT_MAP_CACHE
    ref = pd.read_csv(ref_path)
    mapping = {
        str(name).strip().lower(): district_id
        for name, district_id in zip(ref["district_name"], ref["district_id"])
    }
    _DISTRICT_MAP_CACHE = mapping
    return mapping


def load_modeling_history(modeling_path: str = "data/processed/modeling_dataset.parquet") -> pd.DataFrame:
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None:
        return _HISTORY_CACHE.copy()
    df = _read_parquet_robust(modeling_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["district_id", "timestamp"]).reset_index(drop=True)

    if "peak_demand_mw" in df.columns:
        grp = df.groupby("district_id")["peak_demand_mw"]
        if "demand_lag_24h" not in df.columns:
            df["demand_lag_24h"] = grp.shift(1)
        if "demand_lag_7d" not in df.columns:
            df["demand_lag_7d"] = grp.shift(7)
        if "demand_rolling_24h" not in df.columns:
            df["demand_rolling_24h"] = grp.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        if "demand_rolling_7d" not in df.columns:
            df["demand_rolling_7d"] = grp.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

    for col in ["dayofweek", "month", "is_weekend"]:
        if col not in df.columns:
            if col == "dayofweek":
                df[col] = df["timestamp"].dt.dayofweek
            elif col == "month":
                df[col] = df["timestamp"].dt.month
            else:
                df[col] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df = add_time_features(df)
    df = add_lag_features(df, target="peak_demand_mw")
    _HISTORY_CACHE = df.copy()
    return df


def _build_outage_features(snapshot_df: pd.DataFrame, outages_path: str = "data/processed/outages.parquet") -> pd.DataFrame:
    snap = snapshot_df.copy()
    snap["timestamp"] = pd.to_datetime(snap["timestamp"], errors="coerce")
    outages = _read_parquet_robust(outages_path)
    if outages.empty:
        snap["outages_7d"] = 0.0
        snap["outages_30d"] = 0.0
        return snap

    outages["timestamp"] = pd.to_datetime(outages["timestamp"], errors="coerce")
    name_map = _get_district_name_map()
    outages["district_id"] = outages.get("district", "").astype(str).str.strip().str.lower().map(name_map)
    outages = outages.dropna(subset=["district_id", "timestamp"]).copy()

    outs7, outs30 = [], []
    for _, row in snap.iterrows():
        did = row["district_id"]
        ts = row["timestamp"]
        district_out = outages[outages["district_id"] == did]
        outs7.append(float(((district_out["timestamp"] <= ts) & (district_out["timestamp"] > ts - pd.Timedelta(days=7))).sum()))
        outs30.append(float(((district_out["timestamp"] <= ts) & (district_out["timestamp"] > ts - pd.Timedelta(days=30))).sum()))

    snap["outages_7d"] = outs7
    snap["outages_30d"] = outs30
    return snap


def get_latest_snapshot(modeling_path: str = "data/processed/modeling_dataset.parquet") -> pd.DataFrame:
    df = load_modeling_history(modeling_path)
    latest = df.groupby("district_id").tail(1).reset_index(drop=True).copy()
    latest = _build_outage_features(latest)
    return latest


def _build_feature_frame(row: pd.Series, feature_names) -> pd.DataFrame:
    data = {f: pd.to_numeric(row[f], errors="coerce") if f in row.index else 0.0 for f in feature_names}
    return pd.DataFrame([data]).fillna(0.0)


def predict_demand_on_snapshot(models: Dict[str, object], snapshot_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in snapshot_df.iterrows():
        district = row["district_id"]
        model = models.get(district)
        pred = row.get("peak_demand_mw", np.nan)
        if model is not None:
            feature_names = getattr(model, "feature_names_in_", None)

            if feature_names is None:
                feature_names = sorted([
                    c for c in snapshot_df.columns
                    if c not in {"district_id", "district_name", "timestamp", "Date", "district"}
                ])
            X = _build_feature_frame(row, feature_names)
            try:
                pred = float(model.predict(X)[0])
            except Exception as e:
                print(f"[predictor] model predict failed for {district}: {e}")
        results.append({
            "district_id": district,
            "timestamp": row.get("timestamp", pd.NaT),
            "predicted_peak_demand_mw": float(pred) if pd.notna(pred) else None,
        })
    return pd.DataFrame(results)


def _heuristic_outage_risk(snapshot_df: pd.DataFrame) -> pd.Series:
    snap = snapshot_df.copy()
    demand_z = (snap["peak_demand_mw"] - snap["peak_demand_mw"].mean()) / (snap["peak_demand_mw"].std(ddof=0) + 1e-9)
    out7_z = (snap["outages_7d"] - snap["outages_7d"].mean()) / (snap["outages_7d"].std(ddof=0) + 1e-9)
    out30_z = (snap["outages_30d"] - snap["outages_30d"].mean()) / (snap["outages_30d"].std(ddof=0) + 1e-9)
    pop_z = (pd.to_numeric(snap.get("pop_density", 0), errors="coerce").fillna(0) - pd.to_numeric(snap.get("pop_density", 0), errors="coerce").fillna(0).mean()) / (pd.to_numeric(snap.get("pop_density", 0), errors="coerce").fillna(0).std(ddof=0) + 1e-9)
    score = (
    0.5 * demand_z +
    0.3 * out7_z +
    0.15 * out30_z +
    0.05 * pop_z
)
    return pd.Series(1 / (1 + np.exp(-score)), index=snap.index)


def predict_outage_risk(model_info, snapshot_df: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot_df.copy()
    snap = _build_outage_features(snap)
    probs = None
    if model_info is not None:
        try:
            model = model_info.get("model")
            feat_cols = model_info.get("feature_cols", [])
            X = pd.DataFrame({f: pd.to_numeric(snap.get(f, 0), errors="coerce").fillna(0) for f in feat_cols})
            probs = model.predict_proba(X)[:, 1]
            if np.nanstd(probs) < 1e-9:
                probs = None
        except Exception as e:
            print(f"[predictor] outage model predict_proba failed: {e}")
            probs = None
    if probs is None:
        probs = _heuristic_outage_risk(snap).values
    return pd.DataFrame({
        "district_id": snap["district_id"].values,
        "timestamp": snap["timestamp"].values,
        "outage_risk": probs,
    })


def build_future_feature_frame(modeling_path: str = "data/processed/modeling_dataset.parquet", periods: int = 24, freq: str = "D") -> pd.DataFrame:
    history = load_modeling_history(modeling_path)
    latest = get_latest_snapshot(modeling_path)
    latest_by_district = {did: history[history["district_id"] == did].sort_values("timestamp").copy() for did in history["district_id"].unique()}
    rows = []

    for step in range(1, periods + 1):
        step_rows = []
        for district_id, hist in latest_by_district.items():
            last = hist.iloc[-1].copy()
            ts = pd.Timestamp(last["timestamp"]) + pd.tseries.frequencies.to_offset(freq)
            new_row = last.copy()
            new_row["timestamp"] = ts
            new_row["dayofweek"] = ts.dayofweek
            new_row["month"] = ts.month
            new_row["is_weekend"] = int(ts.dayofweek >= 5)
            new_row["hour"] = ts.hour
            new_row["hour_sin"] = np.sin(2 * np.pi * new_row["hour"] / 24)
            new_row["hour_cos"] = np.cos(2 * np.pi * new_row["hour"] / 24)

            new_row["dow_sin"] = np.sin(2 * np.pi * new_row["dayofweek"] / 7)
            new_row["dow_cos"] = np.cos(2 * np.pi * new_row["dayofweek"] / 7)
            # carry exogenous vars from last available row
            for col in ["temperature_c", "precipitation_mm", "is_hot_day", "is_rainy_day", "pop_density"]:
                if col not in new_row.index:
                    new_row[col] = last.get(col, 0)
            # update lag features using most recent history/predictions
            series = hist["peak_demand_mw"].tolist()
            new_row["demand_lag_24h"] = series[-1] if len(series) >= 1 else last.get("peak_demand_mw", 0)
            new_row["demand_lag_7d"] = series[-7] if len(series) >= 7 else np.mean(series[-min(len(series), 7):])
            window_3 = series[-min(len(series), 3):]
            window_7 = series[-min(len(series), 7):]

            new_row["demand_rolling_24h"] = float(np.mean(window_3))
            new_row["demand_rolling_7d"] = float(np.mean(window_7))


            new_row["demand_std_7d"] = float(np.std(window_7))
            new_row["demand_trend"] = float(window_7[-1] - window_7[0]) if len(window_7) > 1 else 0
            new_row["high_demand_flag"] = int( new_row["peak_demand_mw"] > np.mean(series) )
            step_rows.append(new_row)
        step_df = pd.DataFrame(step_rows)
        preds = predict_demand_on_snapshot(load_demand_models(), step_df)
        pred_map = preds.set_index("district_id")["predicted_peak_demand_mw"].to_dict()
        for district_id, pred in pred_map.items():
            new_row = step_df[step_df["district_id"] == district_id].iloc[0].copy()
            new_row["peak_demand_mw"] = float(pred)
            latest_by_district[district_id] = pd.concat([latest_by_district[district_id], pd.DataFrame([new_row])], ignore_index=True)
            rows.append(new_row)
    future = pd.DataFrame(rows)
    future = _build_outage_features(future)
    return future.reset_index(drop=True)


def predict_snapshot_all(
    model_dir: str = "models/saved",
    modeling_path: str = "data/processed/modeling_dataset.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = load_demand_models(model_dir)
    outage_info = load_outage_model(Path(model_dir) / "outage_model.joblib")
    snap = get_latest_snapshot(modeling_path)
    forecasts = predict_demand_on_snapshot(models, snap)
    risks = predict_outage_risk(outage_info, snap)
    return forecasts, risks


if __name__ == "__main__":
    forecasts, risks = predict_snapshot_all()
    print(forecasts.head())
    print(risks.head())
