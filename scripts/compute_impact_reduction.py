# scripts/compute_impact_reduction_daily.py
"""
Compute baseline vs optimized outage impact (per-day normalization).
Saves outputs/impact_reduction_summary.json and outputs/impact_by_district_debug.csv.

Usage:
    python scripts/compute_impact_reduction_daily.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Paths (adjust if needed)
DIST_REF = Path("data/geographic/ap_districts_reference.csv")
OUTAGES = Path("data/processed/outages.parquet")
SCHEDULE = Path("outputs/load_shedding_schedule.csv")
CRIT = Path("data/processed/critical_infra.parquet")
MODELING = Path("data/processed/modeling_dataset.parquet")
OUT_SUM = Path("outputs/impact_reduction_summary.json")
OUT_CSV = Path("outputs/impact_by_district_debug.csv")

def find_district_col(df):
    for c in ['district_id','district','region_id','region']:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'district' in c.lower():
            return c
    return None

def load_population_map():
    if not DIST_REF.exists():
        return {}
    ref = pd.read_csv(DIST_REF)
    if 'district_id' in ref.columns and 'population' in ref.columns:
        return dict(zip(ref['district_id'].astype(str), pd.to_numeric(ref['population'], errors='coerce').fillna(0.0)))
    # fallback: proxy using pop_density * area_km2
    if all(c in ref.columns for c in ['district_id','pop_density','area_km2']):
        proxy = pd.to_numeric(ref['pop_density'], errors='coerce').fillna(0.0) * pd.to_numeric(ref['area_km2'], errors='coerce').fillna(0.0)
        return dict(zip(ref['district_id'].astype(str), proxy.fillna(0.0)))
    return {}

def load_crit_weights():
    if not CRIT.exists():
        return {}
    cdf = pd.read_parquet(CRIT)
    key = find_district_col(cdf)
    if key is None:
        return {}
    if 'priority_level' in cdf.columns:
        cdf['priority_level'] = pd.to_numeric(cdf['priority_level'], errors='coerce').fillna(0)
        cdf['weight'] = (cdf['priority_level'].max() - cdf['priority_level'] + 1).astype(float)
        agg = cdf.groupby(key)['weight'].sum().to_dict()
    else:
        agg = cdf.groupby(key).size().to_dict()
    return {k: max(1.0, float(v)) for k,v in agg.items()}

def compute_baseline_person_hours(pop_map, crit_w_map):
    total_ph = 0.0
    per_district = {}
    days_span = None
    if not OUTAGES.exists():
        return total_ph, per_district, days_span
    out = pd.read_parquet(OUTAGES)
    dcol = find_district_col(out)
    if dcol is None:
        return total_ph, per_district, days_span
    # durations
    if {'start_timestamp','end_timestamp'}.issubset(out.columns):
        out['start_timestamp'] = pd.to_datetime(out['start_timestamp'], errors='coerce')
        out['end_timestamp'] = pd.to_datetime(out['end_timestamp'], errors='coerce')
        out['duration_hours'] = (out['end_timestamp'] - out['start_timestamp']).dt.total_seconds() / 3600.0
    elif 'duration_hours' in out.columns:
        out['duration_hours'] = pd.to_numeric(out['duration_hours'], errors='coerce').fillna(0.0)
    else:
        out['duration_hours'] = 1.0  # fallback
    # days span estimate
    try:
        ts = pd.to_datetime(out[['start_timestamp','end_timestamp']].stack().dropna().unique())
        if len(ts) >= 2:
            days_span = (ts.max() - ts.min()).days
            if days_span <= 0:
                days_span = None
    except Exception:
        days_span = None
    out[dcol] = out[dcol].astype(str)
    grouped = out.groupby(dcol)['duration_hours'].sum().to_dict()
    for district, hours in grouped.items():
        pop = float(pop_map.get(district, 1.0))
        w = float(crit_w_map.get(district, 1.0))
        ph = pop * hours * w
        per_district[district] = {'outage_hours': float(hours), 'person_hours': float(ph)}
        total_ph += ph
    return total_ph, per_district, days_span

def compute_optimized_person_hours(pop_map, crit_w_map):
    total_ph = 0.0
    per_district = {}
    if not SCHEDULE.exists():
        return total_ph, per_district
    sched = pd.read_csv(SCHEDULE)
    dcol = find_district_col(sched) or ('district_id' if 'district_id' in sched.columns else None)
    hour_cols = [c for c in sched.columns if c.startswith('h') and len(c)>=3]
    if not hour_cols:
        hour_cols = [f'h{h:02d}' for h in range(24) if f'h{h:02d}' in sched.columns]
    for _, row in sched.iterrows():
        district = str(row.get(dcol)) if dcol else None
        hours = 0.0
        if hour_cols:
            hours = sum([float(row.get(h, 0) or 0) for h in hour_cols])
        pop = float(pop_map.get(district, 1.0))
        w = float(crit_w_map.get(district, 1.0))
        ph = pop * hours * w
        per_district[district] = {'scheduled_hours': float(hours), 'person_hours': float(ph)}
        total_ph += ph
    return total_ph, per_district

def main():
    pop_map = load_population_map()
    crit_w_map = load_crit_weights()

    total_baseline_ph, baseline_by, days_span = compute_baseline_person_hours(pop_map, crit_w_map)
    if days_span is None or days_span <= 0:
        # fallback: try modeling dataset span
        if MODELING.exists():
            md = pd.read_parquet(MODELING)
            if 'timestamp' in md.columns:
                ts = pd.to_datetime(md['timestamp'].dropna())
                if not ts.empty:
                    days_span = max(1, (ts.max() - ts.min()).days)
    if days_span is None or days_span <= 0:
        days_span = 1

    baseline_per_day = total_baseline_ph / days_span if days_span else total_baseline_ph

    total_optimized_ph, optimized_by = compute_optimized_person_hours(pop_map, crit_w_map)

    # Summary and percent
    raw_pct = None
    capped_pct = None
    if baseline_per_day > 0:
        raw_pct = 100.0 * (baseline_per_day - total_optimized_ph) / baseline_per_day
        capped_pct = max(-100.0, min(100.0, raw_pct))

    # Print results
    print(f"Baseline totals:\n  total_baseline_person_hours = {total_baseline_ph}")
    print(f"  baseline_per_day = {baseline_per_day} (using days_span={days_span})")
    print(f"Optimized totals:\n  total_optimized_person_hours = {total_optimized_ph}")

    # Top contributors
    base_df = pd.DataFrame([{'district_id':k, 'person_hours':v['person_hours']} for k,v in baseline_by.items()]).sort_values('person_hours', ascending=False) if baseline_by else pd.DataFrame()
    opt_df = pd.DataFrame([{'district_id':k, 'person_hours':v['person_hours']} for k,v in optimized_by.items()]).sort_values('person_hours', ascending=False) if optimized_by else pd.DataFrame()

    print("\nTop baseline contributors (person-hours):")
    if not base_df.empty:
        print(base_df.head(10).to_string(index=False))
    else:
        print("  (none)")

    print("\nTop optimized contributors (person-hours):")
    if not opt_df.empty:
        print(opt_df.head(10).to_string(index=False))
    else:
        print("  (none)")

    # Print schedule hours for top optimized
    if not opt_df.empty:
        print("\nTop optimized districts schedule hours (first 10):")
        for d in opt_df['district_id'].head(10).tolist():
            e = optimized_by.get(d, {})
            print(f"  {d}: hours={e.get('scheduled_hours', 0.0)}, person_hours={e.get('person_hours', 0.0)}")

    print("\nRaw percent change (100*(baseline_per_day - optimized_daily)/baseline_per_day):", raw_pct)
    print("Capped percent change (bounded -100..100):", capped_pct)

    # Save summary + per-district CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for d in sorted(set(list(baseline_by.keys()) + list(optimized_by.keys()))):
        rows.append({
            'district_id': d,
            'baseline_person_hours_total': baseline_by.get(d,{}).get('person_hours', 0.0),
            'optimized_person_hours_daily': optimized_by.get(d,{}).get('person_hours', 0.0),
            'scheduled_hours': optimized_by.get(d,{}).get('scheduled_hours', 0.0),
            'population': pop_map.get(d, None),
            'crit_weight': crit_w_map.get(d,1.0) if 'crit_w_map' in locals() else crit_w_map.get(d,1.0)
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    summary = {
        'baseline_total_person_hours': total_baseline_ph,
        'baseline_daily_person_hours': baseline_per_day,
        'optimized_daily_person_hours': total_optimized_ph,
        'raw_percent_change': raw_pct,
        'capped_percent_change': capped_pct
    }
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUM, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs: {OUT_CSV}, {OUT_SUM}")

if __name__ == "__main__":
    main()