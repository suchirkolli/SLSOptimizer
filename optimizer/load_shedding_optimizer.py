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


def allocate_load_shedding(df: pd.DataFrame, target_reduction_mw: float,
                           max_pct_per_district: float = 0.1,
                           max_hours_per_district: int = 2):
    """
    Allocate load shedding (MW) across districts with a per-district cap,
    and compute a schedule that respects a max number of outage hours per district.
    Returns allocations DataFrame (with shed_mw) and summary dict.
    """
    df = df.copy().sort_values("impact_score")
    allocations = []
    total_shed = 0.0

    for _, row in df.iterrows():
        demand = float(row.get("peak_demand_mw", 0) or 0)
        # maximum MW we allow from this district
        max_shed_mw = demand * max_pct_per_district
        need = target_reduction_mw - total_shed
        if need <= 0:
            shed = 0.0
        else:
            shed = min(max_shed_mw, need)
        allocations.append({
            "district_id": row["district_id"],
            "district_name": row.get("district_name", row["district_id"]),
            "peak_demand_mw": demand,
            "shed_mw": shed,
            "shed_pct": (shed / demand * 100) if demand > 0 else 0.0,
            "impact_score": row["impact_score"]
        })
        total_shed += shed
        if total_shed >= target_reduction_mw:
            # still append remaining districts as zero (so later schedule includes them)
            # continue building allocations for completeness
            pass

    alloc_df = pd.DataFrame(allocations)

    summary = {
        "total_reduction": float(alloc_df["shed_mw"].sum()),
        "target": float(target_reduction_mw)
    }

    # Also compute a suggested schedule based on shed_mw -> hours, respecting max_hours_per_district
    schedule_df, hourly_totals = generate_simple_schedule(alloc_df, time_blocks=24, max_hours_per_district=max_hours_per_district)

    # return allocations and summary, schedule can be created separately if needed
    return alloc_df, summary


def generate_simple_schedule(alloc_df: pd.DataFrame, time_blocks: int = 24, max_hours_per_district: int = 2):
    """
    Generate a binary schedule for each district such that:
      - number of outage hours for district approx = round((shed_mw / peak_demand_mw) * 24)
      - number of hours is clamped to [0, max_hours_per_district]
      - outages are assigned randomly across the day (seeded for reproducibility)
    Returns schedule_df (one row per district) and hourly_totals (array length time_blocks).
    """
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame(), np.zeros(time_blocks)

    np.random.seed(42)
    n = len(alloc_df)
    schedule = np.zeros((n, time_blocks), dtype=float)
    hourly_reduction = []

    # compute hours per district based on proportion of demand
    for i, row in alloc_df.reset_index(drop=True).iterrows():
        demand = float(row.get("peak_demand_mw", 0) or 0)
        shed = float(row.get("shed_mw", 0) or 0)

        if demand > 0 and shed > 0:
            fraction_of_day = shed / demand  # fraction of full outage-equivalent over 24 hours
            hours = int(round(fraction_of_day * time_blocks))
            # clamp hours
            hours = max(0, min(max_hours_per_district, hours))
        else:
            hours = 0

        # if hours=0 but shed>0 and demand>0, ensure at least 1 hour if small shed exists and max_hours_per_district>0
        if hours == 0 and shed > 0 and demand > 0 and max_hours_per_district > 0:
            hours = 1

        # choose hours randomly without replacement
        if hours > 0:
            chosen = np.random.choice(time_blocks, size=hours, replace=False)
            schedule[i, chosen] = 1.0

        # store per-district hourly reduction estimate (for information only)
        per_hour_reduction = shed / max(hours, 1) if hours > 0 else 0.0
        hourly_reduction.append(per_hour_reduction)

    # compute hourly totals (MW) assuming per-district per-hour reduction estimated above
    hourly_totals = np.zeros(time_blocks, dtype=float)
    for i in range(n):
        per_hour = hourly_reduction[i]
        hourly_totals += schedule[i] * per_hour

    # Format schedule_df: columns h00..h23 plus district_id, district_name, total_reduction_mw
    idx = alloc_df.reset_index(drop=True)
    cols = [f"h{h:02d}" for h in range(time_blocks)]
    schedule_df = pd.DataFrame(schedule, columns=cols)
    schedule_df["district_id"] = idx["district_id"].values
    schedule_df["district_name"] = idx.get("district_name", idx["district_id"]).values
    schedule_df["total_reduction_mw"] = idx["shed_mw"].values

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