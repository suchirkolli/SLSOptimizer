"""Load-shedding optimizer with fair, capacity-aware rotation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


_DEFICIT_TOL = 1e-6


def _critical_counts_by_district(critical_infra: pd.DataFrame) -> pd.DataFrame:
    if critical_infra is None or critical_infra.empty:
        return pd.DataFrame(columns=["district_id", "crit_count", "critical_load_mw"])
    df = critical_infra.copy()
    ref = pd.read_csv("data/geographic/ap_districts_reference.csv")
    name_map = {str(n).strip().lower(): did for n, did in zip(ref["district_name"], ref["district_id"])}
    if "district_id" not in df.columns:
        df["district_id"] = df.get("district", "").astype(str).str.strip().str.lower().map(name_map)
    df = df.dropna(subset=["district_id"]).copy()
    if "avg_load_mw" not in df.columns:
        df["avg_load_mw"] = pd.to_numeric(df.get("avg_load_kw", 0), errors="coerce").fillna(0) / 1000.0
    agg = df.groupby("district_id").agg(
        crit_count=("district_id", "size"),
        critical_load_mw=("avg_load_mw", "sum"),
    ).reset_index()
    return agg


def calculate_impact_scores(df: pd.DataFrame, critical_infra: pd.DataFrame | None = None) -> pd.DataFrame:
    scored = df.copy()
    scored["pop_density"] = pd.to_numeric(scored.get("pop_density", 0), errors="coerce").fillna(0)
    scored["outage_risk"] = pd.to_numeric(scored.get("outage_risk", 0), errors="coerce").fillna(0)
    scored["peak_demand_mw"] = pd.to_numeric(scored.get("peak_demand_mw", 0), errors="coerce").fillna(0)
    scored["district_name"] = scored.get("district_name", scored["district_id"])

    crit = _critical_counts_by_district(critical_infra)
    scored = scored.merge(crit, on="district_id", how="left")
    scored["crit_count"] = scored["crit_count"].fillna(0)
    scored["critical_load_mw"] = scored["critical_load_mw"].fillna(0)

    pop_scaled = scored["pop_density"] / max(scored["pop_density"].max(), 1)
    crit_scaled = scored["crit_count"] / max(scored["crit_count"].max(), 1)
    risk_scaled = scored["outage_risk"].clip(0, 1)
    demand_scaled = scored["peak_demand_mw"] / max(scored["peak_demand_mw"].max(), 1)

    scored["impact_score"] = 1.0 + 1.8 * pop_scaled + 2.4 * crit_scaled + 1.6 * risk_scaled + 0.8 * demand_scaled
    scored["shedding_priority"] = 1 / scored["impact_score"]
    return scored


def allocate_load_shedding(
    df: pd.DataFrame,
    target_reduction_mw: float,
    max_pct_per_district: float = 0.30,
    max_hours_per_district: int = 4,
):
    alloc = df.copy().sort_values(["impact_score", "outage_risk", "pop_density"], ascending=[True, True, True]).reset_index(drop=True)
    alloc["base_cap_mw"] = alloc["peak_demand_mw"] * max_pct_per_district
    total_cap = alloc["base_cap_mw"].sum()
    dynamic_pct = max_pct_per_district
    if total_cap + _DEFICIT_TOL < target_reduction_mw:
        dynamic_pct = min(0.95, target_reduction_mw / max(alloc["peak_demand_mw"].sum(), 1e-9) + 0.02)
        alloc["base_cap_mw"] = alloc["peak_demand_mw"] * dynamic_pct

    weights = alloc["shedding_priority"] / alloc["shedding_priority"].sum()
    alloc["shed_mw"] = np.minimum(alloc["base_cap_mw"], weights * target_reduction_mw)
    remaining = target_reduction_mw - alloc["shed_mw"].sum()

    if remaining > _DEFICIT_TOL:
        for idx in alloc.index:
            room = alloc.at[idx, "base_cap_mw"] - alloc.at[idx, "shed_mw"]
            if room <= 0:
                continue
            extra = min(room, remaining)
            alloc.at[idx, "shed_mw"] += extra
            remaining -= extra
            if remaining <= _DEFICIT_TOL:
                break

    alloc["shed_pct"] = np.where(alloc["peak_demand_mw"] > 0, 100 * alloc["shed_mw"] / alloc["peak_demand_mw"], 0)
    alloc["max_hours_per_district"] = max_hours_per_district

    summary = {
        "total_reduction": float(alloc["shed_mw"].sum()),
        "target": float(target_reduction_mw),
        "dynamic_max_pct": float(dynamic_pct * 100),
        "unmet_mw": float(max(target_reduction_mw - alloc["shed_mw"].sum(), 0)),
    }
    return alloc[["district_id", "district_name", "peak_demand_mw", "shed_mw", "shed_pct", "impact_score", "outage_risk", "crit_count"]], summary


def generate_simple_schedule(alloc_df: pd.DataFrame, time_blocks: int = 24, max_hours_per_district: int = 4):
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame(), np.zeros(time_blocks)

    alloc = alloc_df.reset_index(drop=True).copy()
    cols = [f"h{h:02d}" for h in range(time_blocks)]
    schedule = np.zeros((len(alloc), time_blocks), dtype=int)
    hourly_totals = np.zeros(time_blocks, dtype=float)

    order = alloc.sort_values(["shed_pct", "shed_mw"], ascending=False).reset_index()
    start_hour = 0
    for _, row in order.iterrows():
        i = int(row["index"])
        demand = float(alloc.at[i, "peak_demand_mw"])
        shed = float(alloc.at[i, "shed_mw"])
        if demand <= 0 or shed <= 0:
            continue
        hours_needed = int(np.ceil((shed / demand) * time_blocks))
        hours_needed = max(1, min(max_hours_per_district, hours_needed))
        chosen = [(start_hour + j * max(1, time_blocks // max(hours_needed, 1))) % time_blocks for j in range(hours_needed)]
        chosen = sorted(set(chosen))
        while len(chosen) < hours_needed:
            chosen.append((chosen[-1] + 1) % time_blocks if chosen else 0)
            chosen = sorted(set(chosen))
        per_hour = shed / len(chosen)
        for h in chosen[:hours_needed]:
            schedule[i, h] = 1
            hourly_totals[h] += per_hour
        start_hour = (start_hour + 3) % time_blocks

    schedule_df = pd.DataFrame(schedule, columns=cols)
    schedule_df.insert(0, "district_id", alloc["district_id"])
    schedule_df.insert(1, "district_name", alloc["district_name"])
    schedule_df["total_reduction_mw"] = alloc["shed_mw"]
    return schedule_df, hourly_totals
