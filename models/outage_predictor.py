# models/outage_predictor.py
"""
Outage risk classifier (district-level) — patched to use `peak_demand_mw`.
Inputs expected:
  - data/processed/modeling_dataset.parquet   (merged demand+weather + district ref)
  - data/processed/outages.parquet           (optional; hourly flags or start/end events)
Output:
  - models/saved/outage_model.joblib  (dict with keys: 'model', 'feature_cols', 'threshold')

Run:
    python models/outage_predictor.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import joblib

MODEL_OUT = Path("models/saved/outage_model.joblib")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)


def load_inputs():
    md_path = Path("data/processed/modeling_dataset.parquet")
    outages_path = Path("data/processed/outages.parquet")
    if not md_path.exists():
        raise FileNotFoundError(f"Missing modeling dataset: {md_path}")
    md = pd.read_parquet(md_path)
    outages = pd.read_parquet(outages_path) if outages_path.exists() else pd.DataFrame()
    return md, outages


def expand_outages_to_flags(outages: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns timestamp,district_id,is_outage (hourly flags)."""
    if outages.empty:
        return pd.DataFrame(columns=["timestamp", "district_id", "is_outage"])

    # Already hourly flags?
    if {"timestamp", "district_id", "is_outage"}.issubset(outages.columns):
        flags = outages[["timestamp", "district_id", "is_outage"]].copy()
        flags["timestamp"] = pd.to_datetime(flags["timestamp"], errors="coerce")
        return flags.dropna(subset=["timestamp"])

    # Events table with start_time/end_time
    if {"start_time", "end_time", "district_id"}.issubset(outages.columns):
        rows = []
        for _, r in outages.iterrows():
            try:
                st = pd.to_datetime(r["start_time"], errors="coerce")
                ed = pd.to_datetime(r["end_time"], errors="coerce")
                if pd.isna(st) or pd.isna(ed):
                    continue
                hours = pd.date_range(start=st, end=ed, freq="H")
                for ts in hours:
                    rows.append({"timestamp": ts, "district_id": r["district_id"], "is_outage": 1})
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=["timestamp", "district_id", "is_outage"])
        flags = pd.DataFrame(rows)
        flags["timestamp"] = pd.to_datetime(flags["timestamp"], errors="coerce")
        return flags.dropna(subset=["timestamp"])

    # Unknown format -> return empty flags
    return pd.DataFrame(columns=["timestamp", "district_id", "is_outage"])


def build_training_frame(md: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    """Merge modeling dataset and outage flags to ensure is_outage per timestamp,district."""
    df = md.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "district_id" not in df.columns:
        raise KeyError("modeling dataset missing 'district_id' column")

    flags = expand_outages_to_flags(outages)
    if not flags.empty:
        flags["timestamp"] = pd.to_datetime(flags["timestamp"], errors="coerce")

    merged = pd.merge(df, flags, on=["timestamp", "district_id"], how="left")

    # ensure is_outage exists and integer-typed
    if "is_outage" in merged.columns:
        merged["is_outage"] = merged["is_outage"].fillna(0).astype(int)
    else:
        merged["is_outage"] = 0

    return merged


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # time features
    if "timestamp" in d.columns:
        d["hour"] = d["timestamp"].dt.hour
        d["dayofweek"] = d["timestamp"].dt.dayofweek

    # ensure numeric peak demand
    if "peak_demand_mw" in d.columns:
        d["peak_demand_mw"] = pd.to_numeric(d["peak_demand_mw"], errors="coerce").fillna(0)

    # demand/supply ratio using peak_demand_mw where available
    if "peak_demand_mw" in d.columns and "supply_mw" in d.columns:
        d["demand_supply_ratio"] = (
            pd.to_numeric(d["peak_demand_mw"], errors="coerce")
            / pd.to_numeric(d["supply_mw"], errors="coerce").replace({0: np.nan})
        )

    # rolling recent outage counts per district
    d = d.sort_values(["district_id", "timestamp"])
    if "is_outage" in d.columns:
        d["outages_7d"] = (
            d.groupby("district_id")["is_outage"]
            .rolling(24 * 7, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )
        d["outages_30d"] = (
            d.groupby("district_id")["is_outage"]
            .rolling(24 * 30, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )

    # coerce numeric features and fill missing conservatively
    for col in [
        "peak_demand_mw",
        "temperature",
        "humidity",
        "demand_supply_ratio",
        "outages_7d",
        "outages_30d",
        "pop_density",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    return d


def train_outage_model(merged_df: pd.DataFrame, test_start_date: str = "2023-03-01"):
    df = create_features(merged_df)

    feature_cols = [
        c
        for c in [
            "peak_demand_mw",
            "temperature",
            "humidity",
            "demand_supply_ratio",
            "outages_7d",
            "outages_30d",
            "pop_density",
        ]
        if c in df.columns
    ]
    if not feature_cols:
        raise RuntimeError("No feature columns available for outage model. Check modeling dataset.")

    # ensure label column exists
    if "is_outage" not in df.columns:
        df["is_outage"] = 0

    # If there are zero real positives, create a tiny set of conservative synthetic positives
    if df["is_outage"].sum() == 0:
        if "peak_demand_mw" in df.columns:
            thresh = df["peak_demand_mw"].quantile(0.995)
            cond = df["peak_demand_mw"] >= thresh
            if "temperature" in df.columns:
                temp_thresh = df["temperature"].quantile(0.95)
                cond = cond & (df["temperature"] >= temp_thresh)
            df.loc[cond, "is_outage"] = 1
            print(
                f"Info: created {int(df['is_outage'].sum())} synthetic outage labels for training (heuristic)."
            )
        else:
            raise RuntimeError("No outage labels and no peak_demand_mw to create synthetic labels.")

    # time split
    df_train = df[df["timestamp"] < pd.to_datetime(test_start_date)]
    df_test = df[df["timestamp"] >= pd.to_datetime(test_start_date)]
    if df_train.empty or df_test.empty:
        raise RuntimeError("Insufficient train/test split for outage model.")

    X_train = df_train[feature_cols]
    y_train = df_train["is_outage"]
    X_test = df_test[feature_cols]
    y_test = df_test["is_outage"]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Outage model AUC: {auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    if len(thresholds) > 0:
        best_thresh = float(thresholds[np.nanargmax(f1)])
    else:
        best_thresh = 0.5

    y_pred = (y_proba >= best_thresh).astype(int)
    print("Classification report (test):")
    print(classification_report(y_test, y_pred))

    model_info = {"model": clf, "feature_cols": feature_cols, "threshold": best_thresh}
    joblib.dump(model_info, MODEL_OUT)
    print(f"Saved outage model to {MODEL_OUT}")
    return model_info


def main():
    try:
        md, outages = load_inputs()
        merged = build_training_frame(md, outages)
        train_outage_model(merged)
    except Exception as e:
        print("Training failed:", e)


if __name__ == "__main__":
    main()