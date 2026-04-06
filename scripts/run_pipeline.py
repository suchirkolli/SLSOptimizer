"""End-to-end pipeline runner for Smart Load Shedding Optimizer."""

import os
import sys
import argparse
from pathlib import Path
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.predictor import predict_snapshot_all
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


def prepare_input_for_optimizer(forecasts: pd.DataFrame, risks: pd.DataFrame, district_ref_path: Path) -> pd.DataFrame:
    df = forecasts.merge(risks[["district_id", "outage_risk"]], on="district_id", how="left")
    if "predicted_peak_demand_mw" in df.columns:
        df["peak_demand_mw"] = pd.to_numeric(df["predicted_peak_demand_mw"], errors="coerce")
    if district_ref_path.exists():
        ref = pd.read_csv(district_ref_path)
        keep = [c for c in ["district_id", "district_name", "pop_density"] if c in ref.columns]
        df = df.merge(ref[keep], on="district_id", how="left")
    df["peak_demand_mw"] = pd.to_numeric(df.get("peak_demand_mw", 0), errors="coerce").fillna(0)
    df["outage_risk"] = pd.to_numeric(df.get("outage_risk", 0), errors="coerce").fillna(0)
    return df


def run(deficit_mw: float, out_dir: str = "outputs", use_models: bool = True):
    LOG.info("Starting pipeline: use_models=%s, deficit=%.1f MW", use_models, deficit_mw)
    forecasts, risks = predict_snapshot_all()
    LOG.info("Forecasts rows: %d; Risks rows: %d", len(forecasts), len(risks))
    input_df = prepare_input_for_optimizer(forecasts, risks, Path("data/geographic/ap_districts_reference.csv"))
    crit_path = Path("data/processed/critical_infra.parquet")
    critical = pd.read_parquet(crit_path) if crit_path.exists() else pd.DataFrame()
    scored = calculate_impact_scores(input_df, critical)
    allocations, summary = allocate_load_shedding(scored, float(deficit_mw), max_pct_per_district=0.30, max_hours_per_district=4)
    schedule_df, hourly_totals = generate_simple_schedule(allocations, max_hours_per_district=4)
    LOG.info("Allocation complete. total_reduction=%.2f target=%.2f", summary["total_reduction"], summary["target"])
    save_outputs(Path(out_dir), forecasts, risks, allocations, schedule_df)
    return forecasts, risks, allocations, schedule_df


def parse_args():
    p = argparse.ArgumentParser(description="Run SLS Optimizer pipeline: predict + optimize")
    p.add_argument("--deficit", type=float, default=1500.0, help="Target reduction (MW)")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--use-models", action="store_true", help="Retained for compatibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(deficit_mw=args.deficit, out_dir=args.out_dir, use_models=True)
