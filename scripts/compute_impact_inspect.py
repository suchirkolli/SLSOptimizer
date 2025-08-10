# scripts/compute_impact_inspect.py
"""
Inspect baseline vs optimized impact and show per-district diagnostics.
Run from project root:
    python scripts/compute_impact_inspect.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Paths (adjust if needed)
DIST_REF = Path("data/geographic/ap_districts_reference.csv")
OUTAGES = Path("data/processed/outages.parquet")
SCHEDULE = Path("outputs/load_shedding_schedule.csv")
ALLOC = Path("outputs/load_shedding_allocations.csv")
CRIT = Path("data/processed/critical_infra.parquet")
OUT_CSV = Path("outputs/impact_by_district_debug.csv")
OUT_SUM = Path("outputs/impact_inspect_summary.json")

def find_district_col(df):
    for c in ['district_id','district','region_id','region','districtName']:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'district' in c.lower():
            return c
    return None

# Load district reference and populations/proxies
pop_map = {}
if DIST_REF.exists():
    ref = pd.read_csv(DIST_REF)
    print("District reference columns:", ref.columns.tolist())
    if 'district_id' in ref.columns and 'population' in ref.columns:
        pop_map = dict(zip(ref['district_id'].astype(str), pd.to_numeric(ref['population'], errors='coerce').fillna(0.0)))
    elif 'district_id' in ref.columns and 'pop_density' in ref.columns and 'area_km2' in ref.columns:
        proxy = pd.to_numeric(ref['pop_density'], errors='coerce').fillna(0.0) * pd.to_numeric(ref['area_km2'], errors='coerce').fillna(0.0)
        pop_map = dict(zip(ref['district_id'].astype(str), proxy.fillna(0.0)))
        print("Using pop_density * area_km2 as population proxy.")
    else:
        print("No population found in district ref; fallback pop=1 will be used.")
else:
    print("District reference missing; pop map empty (fallback used).")

# critical infra weights
crit_weight = {}
if CRIT.exists():
    cdf = pd.read_parquet(CRIT)
    key = find_district_col(cdf)
    print("Critical infra columns:", cdf.columns.tolist())
    if key:
        # compute simple weight
        if 'priority_level' in cdf.columns:
            cdf['priority_level'] = pd.to_numeric(cdf['priority_level'], errors='coerce').fillna(0)
            cdf['weight'] = (cdf['priority_level'].max() - cdf['priority_level'] + 1).astype(float)
            crit_agg = cdf.groupby(key)['weight'].sum().to_dict()
        else:
            crit_agg = cdf.groupby(key).size().to_dict()
        crit_weight = {k: max(1.0, float(v)) for k,v in crit_agg.items()}
    else:
        print("No district column in critical infra; skipping weights.")
else:
    print("No critical infra file found; using weight=1 for all districts.")

def crit_w(d): return float(crit_weight.get(d, 1.0))

# Baseline: compute person-hours per district and overall and days span
baseline_by = {}
total_baseline_ph = 0.0
days_span = None
if OUTAGES.exists():
    out = pd.read_parquet(OUTAGES)
    dcol = find_district_col(out)
    print("Outages columns:", out.columns.tolist())
    if dcol:
        # get durations
        if {'start_timestamp','end_timestamp'}.issubset(out.columns):
            out['start_timestamp'] = pd.to_datetime(out['start_timestamp'], errors='coerce')
            out['end_timestamp'] = pd.to_datetime(out['end_timestamp'], errors='coerce')
            out['duration_hours'] = (out['end_timestamp'] - out['start_timestamp']).dt.total_seconds() / 3600.0
        elif 'duration_hours' in out.columns:
            out['duration_hours'] = pd.to_numeric(out['duration_hours'], errors='coerce').fillna(0.0)
        else:
            out['duration_hours'] = 1.0
        # days span estimate
        ts = pd.to_datetime(out[['start_timestamp','end_timestamp']].stack().dropna().unique())
        if len(ts) >= 2:
            days_span = (ts.max() - ts.min()).days
        out[dcol] = out[dcol].astype(str)
        grouped = out.groupby(dcol)['duration_hours'].sum().to_dict()
        for d, hours in grouped.items():
            pop = pop_map.get(str(d), 1.0)
            w = crit_w(str(d))
            ph = float(pop) * float(hours) * float(w)
            baseline_by[str(d)] = {'outage_hours': float(hours), 'person_hours': ph}
            total_baseline_ph += ph
    else:
        print("Outages present but no district column found; baseline empty.")
else:
    print("No outages.parquet found; baseline empty.")

if days_span is None or days_span <= 0:
    # try modeling dataset span
    mdp = Path("data/processed/modeling_dataset.parquet")
    if mdp.exists():
        md = pd.read_parquet(mdp)
        if 'timestamp' in md.columns:
            ts = pd.to_datetime(md['timestamp'].dropna())
            if not ts.empty:
                days_span = (ts.max() - ts.min()).days
if days_span is None or days_span <= 0:
    days_span = 1
print(f"Using days_span = {days_span} for baseline per-day normalization.")

baseline_per_day = total_baseline_ph / days_span if days_span else total_baseline_ph

# Optimized: compute scheduled person-hours (per schedule)
optimized_by = {}
total_optimized_ph = 0.0
if SCHEDULE.exists():
    sched = pd.read_csv(SCHEDULE)
    sched_key = find_district_col(sched) or ('district_id' if 'district_id' in sched.columns else None)
    hour_cols = [c for c in sched.columns if c.startswith('h') and len(c)>=3]
    if not hour_cols:
        hour_cols = [f'h{h:02d}' for h in range(24) if f'h{h:02d}' in sched.columns]
    for _, r in sched.iterrows():
        d = str(r.get(sched_key)) if sched_key else None
        hours = sum([float(r.get(h,0) or 0) for h in hour_cols]) if hour_cols else 0.0
        pop = pop_map.get(d, 1.0)
        w = crit_w(d)
        ph = float(pop) * float(hours) * float(w)
        optimized_by[d] = {'scheduled_hours': hours, 'person_hours': ph}
        total_optimized_ph += ph
else:
    print("No schedule file found; optimized empty.")

# Print diagnostics
print("\nBaseline totals:")
print(f"  total_baseline_person_hours = {total_baseline_ph}")
print(f"  baseline_per_day = {baseline_per_day} (using days_span={days_span})")
print("\nOptimized totals:")
print(f"  total_optimized_person_hours = {total_optimized_ph}")

# Show top per-district contributors for baseline and optimized
base_df = pd.DataFrame([{'district_id':k, **v} for k,v in baseline_by.items()]).sort_values('person_hours', ascending=False) if baseline_by else pd.DataFrame()
opt_df = pd.DataFrame([{'district_id':k, **v} for k,v in optimized_by.items()]).sort_values('person_hours', ascending=False) if optimized_by else pd.DataFrame()

print("\nTop baseline contributors (person-hours):")
print(base_df[['district_id','person_hours']].head(10).to_string(index=False))

print("\nTop optimized contributors (person-hours):")
print(opt_df[['district_id','person_hours']].head(10).to_string(index=False))

# Print schedule hours for top optimized districts
if not opt_df.empty:
    print("\nTop optimized districts schedule hours (first 10):")
    for d in opt_df['district_id'].head(10).tolist():
        entry = optimized_by.get(d, {})
        print(f"  {d}: hours={entry.get('scheduled_hours')}, person_hours={entry.get('person_hours')}")

# Print suspicious checks
print("\nChecks:")
# 1) any district with scheduled_hours > 24?
overs = [d for d,v in optimized_by.items() if v.get('scheduled_hours',0) > 24]
print("  districts with scheduled_hours > 24:", overs)
# 2) any district with extremely large population
large_pops = {d:p for d,p in pop_map.items() if p>50_000_000}  # >50M unrealistic
print("  districts with pop >50M (suspicious):", large_pops)

# Save per-district CSV for review
rows = []
for d in sorted(set(list(baseline_by.keys()) + list(optimized_by.keys()))):
    rows.append({
        'district_id': d,
        'baseline_person_hours': baseline_by.get(d,{}).get('person_hours', 0.0),
        'optimized_person_hours': optimized_by.get(d,{}).get('person_hours', 0.0),
        'scheduled_hours': optimized_by.get(d,{}).get('scheduled_hours', 0.0),
        'population': pop_map.get(d, None),
        'crit_weight': crit_weight.get(d,1.0)
    })
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
summary = {
    'total_baseline_person_hours': total_baseline_ph,
    'baseline_per_day': baseline_per_day,
    'total_optimized_person_hours': total_optimized_ph
}
OUT_SUM.parent.mkdir(exist_ok=True)
with open(OUT_SUM, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved per-district CSV to {OUT_CSV} and summary to {OUT_SUM}")