# optimizer/load_shedding_optimizer.py
"""
Basic rule-based load-shedding optimizer (district-level), using `peak_demand_mw`.
Saves allocations and a simple rotational schedule to outputs/.
Usage:
    python -c "from optimizer.load_shedding_optimizer import main_demo; main_demo(1500.0)"
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

MODEL_OUT = Path("models/saved/outage_model.joblib")


def calculate_impact_scores(df: pd.DataFrame, critical_infra: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    df["pop_density"] = pd.to_numeric(df.get("pop_density", 0), errors="coerce").fillna(0)
    df["impact_base"] = df["pop_density"].replace({0: 1.0}).astype(float)

    # incorporate outage risk if present
    if "outage_risk" in df.columns:
        df["impact_score"] = df["impact_base"] * (1 + df["outage_risk"])
    else:
        df["impact_score"] = df["impact_base"]

    # penalize districts with critical infrastructure (guard against missing column)
    if critical_infra is not None and not critical_infra.empty and "district_id" in critical_infra.columns:
        crit_counts = critical_infra.groupby("district_id").size().rename("crit_count").reset_index()
        df = df.merge(crit_counts, on="district_id", how="left")
        df["crit_count"] = df["crit_count"].fillna(0)
        df["impact_score"] = df["impact_score"] * (1 + 0.25 * df["crit_count"])
    else:
        # ensure column exists for downstream code
        df["crit_count"] = 0

    return df


def allocate_load_shedding(df: pd.DataFrame, target_reduction_mw: float, max_pct_per_district: float = 0.3):
    df = df.copy().sort_values("impact_score")
    allocations = []
    total_shed = 0.0

    for _, row in df.iterrows():
        # read peak_demand_mw as the demand metric
        demand = float(row.get("peak_demand_mw", 0) or 0)
        max_shed = demand * max_pct_per_district
        need = target_reduction_mw - total_shed
        shed = 0.0 if need <= 0 else min(max_shed, need)
        allocations.append(
            {
                "district_id": row["district_id"],
                "district_name": row.get("district_name", row["district_id"]),
                "peak_demand_mw": demand,
                "shed_mw": shed,
                "shed_pct": (shed / demand * 100) if demand > 0 else 0.0,
                "impact_score": row["impact_score"],
            }
        )
        total_shed += shed
        if total_shed >= target_reduction_mw:
            break

    alloc_df = pd.DataFrame(allocations)
    summary = {"total_reduction": float(alloc_df["shed_mw"].sum()), "target": float(target_reduction_mw)}
    return alloc_df, summary


def generate_simple_schedule(alloc_df: pd.DataFrame, time_blocks: int = 24):
    n = len(alloc_df)
    if n == 0:
        return pd.DataFrame(), np.zeros(time_blocks)
    np.random.seed(42)
    schedule = np.zeros((n, time_blocks))
    hourly_reduction = alloc_df["shed_mw"].values / time_blocks
    for i in range(n):
        hours = np.random.choice(time_blocks, size=max(1, time_blocks // 3), replace=False)
        schedule[i, hours] = 1
    schedule_df = pd.DataFrame(schedule, index=alloc_df["district_id"], columns=[f"h{h:02d}" for h in range(time_blocks)])
    schedule_df["district_id"] = alloc_df["district_id"].values
    schedule_df["district_name"] = alloc_df["district_name"].values
    schedule_df["total_reduction_mw"] = alloc_df["shed_mw"].values
    hourly_totals = (schedule * hourly_reduction[:, None]).sum(axis=0)
    return schedule_df, hourly_totals


def main_demo(target_reduction_mw: float = 1500.0):
    md_path = Path("data/processed/modeling_dataset.parquet")
    ref_path = Path("data/geographic/ap_districts_reference.csv")
    crit_path = Path("data/processed/critical_infra.parquet")

    if not md_path.exists():
        raise FileNotFoundError(md_path)
    md = pd.read_parquet(md_path)

    # use last timestamp per district as "current"
    md = md.sort_values("timestamp")
    latest = md.groupby("district_id").tail(1)

    # make an explicit copy before modifying to avoid SettingWithCopyWarning
    latest = latest.copy()

    # ensure peak_demand_mw exists
    if "peak_demand_mw" not in latest.columns:
        raise RuntimeError("modeling dataset lacks 'peak_demand_mw' column - run power distributor first")

    district_ref = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
    critical = pd.read_parquet(crit_path) if crit_path.exists() else pd.DataFrame()

    # prepare input frame using peak_demand_mw
    input_df = latest[["district_id", "peak_demand_mw"]].copy()
    if "district_name" in latest.columns:
        input_df = input_df.merge(latest[["district_id", "district_name"]].drop_duplicates(), on="district_id", how="left")
    if not district_ref.empty:
        input_df = input_df.merge(district_ref[["district_id", "pop_density", "district_name"]].drop_duplicates(), on="district_id", how="left")

    # outage risk: use model if available
    if MODEL_OUT.exists():
        mi = joblib.load(MODEL_OUT)
        feat_cols = mi["feature_cols"]
        # ensure features present in latest; set missing ones to 0 using .loc
        for c in feat_cols:
            if c not in latest.columns:
                latest.loc[:, c] = 0
        X = latest[feat_cols].fillna(0)
        risks = mi["model"].predict_proba(X)[:, 1]
        risk_df = pd.DataFrame({"district_id": latest["district_id"].values, "outage_risk": risks})
        input_df = input_df.merge(risk_df, on="district_id", how="left")
    else:
        # fallback: normalized outages_7d if present else zeros
        input_df["outage_risk"] = latest.get("outages_7d", 0).fillna(0).astype(float)
        if input_df["outage_risk"].max() > 0:
            input_df["outage_risk"] = input_df["outage_risk"] / (input_df["outage_risk"].max() + 1e-9)

    scored = calculate_impact_scores(input_df, critical)
    alloc_df, summary = allocate_load_shedding(scored, float(target_reduction_mw))
    schedule_df, hourly_totals = generate_simple_schedule(alloc_df)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    alloc_df.to_csv(out_dir / "load_shedding_allocations.csv", index=False)
    schedule_df.to_csv(out_dir / "load_shedding_schedule.csv", index=False)

    print("Allocations (top):")
    print(alloc_df.sort_values("shed_mw", ascending=False).to_string(index=False))
    print("\nSummary:", summary)
    return alloc_df, schedule_df, summary


if __name__ == "__main__":
    main_demo(1500.0)