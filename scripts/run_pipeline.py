# scripts/run_pipeline.py
"""
End-to-end pipeline runner for Smart Load Shedding Optimizer.

Usage (from project root):
  python scripts/run_pipeline.py --deficit 1500 --use-models

This script ensures the project root is on sys.path so `from services...` imports work
whether you run the script from the repo root or from the scripts/ directory.
"""
import os
import sys
# Ensure project root (parent of scripts/) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
import logging
import pandas as pd

# Import project modules (now that sys.path is set)
from services.predictor import predict_snapshot_all, get_latest_snapshot
from optimizer.load_shedding_optimizer import calculate_impact_scores, allocate_load_shedding, generate_simple_schedule

LOG = logging.getLogger("run_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_outputs(out_dir: Path, forecasts: pd.DataFrame, risks: pd.DataFrame, allocations: pd.DataFrame, schedule: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_dir / "forecast.csv", index=False)
    risks.to_csv(out_dir / "outage_risk.csv", index=False)
    allocations.to_csv(out_dir / "load_shedding_allocations.csv", index=False)
    schedule.to_csv(out_dir / "load_shedding_schedule.csv", index=False)
    LOG.info("Saved outputs to %s", out_dir)


def prepare_input_for_optimizer(forecasts: pd.DataFrame, risks: pd.DataFrame, district_ref_path: Path, use_predicted: bool = True) -> pd.DataFrame:
    df = forecasts.merge(risks[['district_id', 'outage_risk']], on='district_id', how='left')
    # unify demand column names
    if use_predicted and 'predicted_peak_demand_mw' in df.columns:
        df = df.rename(columns={'predicted_peak_demand_mw': 'peak_demand_mw'})
    # fallback candidates
    for cand in ['peak_demand_mw', 'predicted_peak_demand_mw', 'demand_mw']:
        if cand in df.columns:
            df = df.rename(columns={cand: 'peak_demand_mw'})
            break
    # attach district reference (pop_density, district_name) if available
    if district_ref_path.exists():
        ref = pd.read_csv(district_ref_path)
        keep = [c for c in ['district_id', 'pop_density', 'district_name'] if c in ref.columns]
        if keep:
            df = df.merge(ref[keep], on='district_id', how='left')
    else:
        LOG.warning("District reference not found at %s", district_ref_path)
    # ensure numeric columns
    df['peak_demand_mw'] = pd.to_numeric(df.get('peak_demand_mw', 0), errors='coerce').fillna(0.0)
    df['outage_risk'] = pd.to_numeric(df.get('outage_risk', 0), errors='coerce').fillna(0.0)
    return df


def run(deficit_mw: float, out_dir: str = "outputs", use_models: bool = True):
    out_path = Path(out_dir)
    LOG.info("Starting pipeline: use_models=%s, deficit=%.1f MW", use_models, deficit_mw)

    # 1) Forecasts & risks
    try:
        if use_models:
            forecasts, risks = predict_snapshot_all()
        else:
            snap = get_latest_snapshot()
            if 'peak_demand_mw' in snap.columns:
                forecasts = snap[['district_id', 'timestamp', 'peak_demand_mw']].rename(columns={'peak_demand_mw': 'predicted_peak_demand_mw'})
            else:
                forecasts = pd.DataFrame([{'district_id': r, 'timestamp': pd.NaT, 'predicted_peak_demand_mw': 0.0} for r in snap['district_id'].unique()])
            risks = pd.DataFrame({'district_id': forecasts['district_id'].values, 'timestamp': forecasts['timestamp'].values, 'outage_risk': 0.0})
    except Exception as e:
        LOG.exception("Prediction step failed: %s", e)
        raise

    LOG.info("Forecasts rows: %d; Risks rows: %d", len(forecasts), len(risks))

    # 2) Prepare input for optimizer
    district_ref_path = Path("data/geographic/ap_districts_reference.csv")
    input_df = prepare_input_for_optimizer(forecasts, risks, district_ref_path, use_predicted=True)

    # 3) Load critical infra if available
    crit_path = Path("data/processed/critical_infra.parquet")
    critical = pd.read_parquet(crit_path) if crit_path.exists() else pd.DataFrame()

    # 4) Calculate impact scores and allocate
    scored = calculate_impact_scores(input_df, critical)
    allocations, summary = allocate_load_shedding(scored, float(deficit_mw))
    schedule_df, hourly_totals = generate_simple_schedule(allocations)

    LOG.info("Allocation complete. total_reduction=%.2f target=%.2f", summary['total_reduction'], summary['target'])

    # 5) Save outputs
    save_outputs(out_path, forecasts, risks, allocations, schedule_df)
    LOG.info("Pipeline finished. Summary: %s", summary)
    return forecasts, risks, allocations, schedule_df


def parse_args():
    p = argparse.ArgumentParser(description="Run SLS Optimizer pipeline: predict + optimize")
    p.add_argument("--deficit", type=float, default=1500.0, help="Target reduction (MW)")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--use-models", action="store_true", help="Use saved models for forecasts/risks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # auto-detect models if not specified
    use_models_flag = args.use_models
    if not use_models_flag:
        models_dir = Path("models/saved")
        if models_dir.exists() and any(models_dir.glob("demand_model_*.joblib")):
            use_models_flag = True
    try:
        run(deficit_mw=args.deficit, out_dir=args.out_dir, use_models=use_models_flag)
    except Exception as e:
        LOG.exception("Pipeline failed: %s", e)
        raise