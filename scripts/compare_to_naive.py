# scripts/compare_to_naive.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Config: tune if needed
MODELING = Path("data/processed/modeling_dataset.parquet")
DIST_REF = Path("data/geographic/ap_districts_reference.csv")
OPT_ALLOC = Path("outputs/load_shedding_allocations.csv")
OPT_SCHED = Path("outputs/load_shedding_schedule.csv")
CRIT = Path("data/processed/critical_infra.parquet")
OUT_SUM = Path("outputs/compare_naive_summary.json")

# Parameters that must match optimizer's hours rule
TIME_BLOCKS = 24
MAX_HOURS = 2   # same cap used in optimizer
MAX_PCT = 0.10  # used when optimizer run; not needed here except sanity

def find_district_col(df):
    for c in ['district_id','district','region_id','region']:
        if c in df.columns: return c
    for c in df.columns:
        if 'district' in c.lower(): return c
    return None

# Load data
if not MODELING.exists():
    raise SystemExit("Missing modeling dataset: run validate_data.py and pipeline first")
snap = pd.read_parquet(MODELING).sort_values(['district_id','timestamp']).groupby('district_id').tail(1)
snap = snap.reset_index(drop=True)

# population map
pop_map = {}
if DIST_REF.exists():
    dref = pd.read_csv(DIST_REF)
    if 'district_id' in dref.columns and 'population' in dref.columns:
        pop_map = dict(zip(dref['district_id'].astype(str), pd.to_numeric(dref['population'], errors='coerce').fillna(0.0)))
    else:
        # fallback: try pop_density * area_km2
        if all(c in dref.columns for c in ['district_id','pop_density','area_km2']):
            proxy = (pd.to_numeric(dref['pop_density'],errors='coerce').fillna(0) * pd.to_numeric(dref['area_km2'],errors='coerce').fillna(0))
            pop_map = dict(zip(dref['district_id'].astype(str), proxy.fillna(0.0)))
        else:
            # fallback 1
            pop_map = {d:1.0 for d in snap['district_id'].astype(str).unique()}

# crit weight
crit_weight = {}
if CRIT.exists():
    cdf = pd.read_parquet(CRIT)
    k = find_district_col(cdf)
    if k:
        if 'priority_level' in cdf.columns:
            cdf['priority_level'] = pd.to_numeric(cdf['priority_level'], errors='coerce').fillna(0)
            cdf['weight'] = (cdf['priority_level'].max() - cdf['priority_level'] + 1).astype(float)
            crit_weight = cdf.groupby(k)['weight'].sum().to_dict()
        else:
            crit_weight = cdf.groupby(k).size().to_dict()
# ensure >=1
crit_weight = {k:max(1.0,float(v)) for k,v in crit_weight.items()}

# sum peak to get total capacity available
snap['peak'] = pd.to_numeric(snap.get('peak_demand_mw', snap.get('predicted_peak_demand_mw', 0)), errors='coerce').fillna(0)
sum_peak = snap['peak'].sum()
if sum_peak <= 0:
    raise SystemExit("No peak_demand_mw values found in snapshot")

# Load optimizer allocations & schedule
if not OPT_ALLOC.exists() or not OPT_SCHED.exists():
    raise SystemExit("Missing optimizer outputs in outputs/; run pipeline first")

opt_alloc = pd.read_csv(OPT_ALLOC)
opt_sched = pd.read_csv(OPT_SCHED)

total_target = opt_alloc['shed_mw'].sum()
print("Total optimized target MW (from alloc file):", total_target)

# Build naive allocation proportional to peak
weights = snap.set_index('district_id')['peak'].to_dict()
districts = list(snap['district_id'].astype(str))
w_arr = np.array([weights.get(d,0) for d in districts], dtype=float)
w_sum = w_arr.sum()
if w_sum == 0:
    w_arr = np.ones(len(districts)) / len(districts)
else:
    w_arr = w_arr / w_sum

naive_shed = {d: float(w_arr[i] * total_target) for i,d in enumerate(districts)}

# function to convert shed_mw to hours (same rule optimizer used)
def shed_to_hours(shed_mw, peak_mw, time_blocks=TIME_BLOCKS, max_hours=MAX_HOURS):
    if peak_mw <= 0:
        return 0
    frac = shed_mw / peak_mw
    hours = int(round(frac * time_blocks))
    hours = max(0, min(max_hours, hours))
    if hours == 0 and shed_mw>0 and max_hours>0:
        hours = 1
    return hours

# compute person-hours for naive and optimized
naive_ph = 0.0
opt_ph = 0.0
naive_by = {}
opt_by = {}

# optimized: derive schedule hours from schedule file (sum h00..h23)
hour_cols = [c for c in opt_sched.columns if c.startswith('h')]
for _, r in opt_sched.iterrows():
    d = str(r.get('district_id', r.get('district', r.get('region_id', None))))
    hours = sum([float(r.get(h,0) or 0) for h in hour_cols]) if hour_cols else 0.0
    pop = float(pop_map.get(d,1.0))
    w = float(crit_weight.get(d,1.0))
    ph = pop * hours * w
    opt_by[d] = {'hours':hours, 'person_hours':ph}
    opt_ph += ph

# naive: convert naive_shed to hours using snapshot peaks
for d, shed in naive_shed.items():
    peak = float(snap.loc[snap['district_id']==d, 'peak'].iloc[0]) if not snap[snap['district_id']==d].empty else 0.0
    hours = shed_to_hours(shed, peak, TIME_BLOCKS, MAX_HOURS)
    pop = float(pop_map.get(d,1.0))
    w = float(crit_weight.get(d,1.0))
    ph = pop * hours * w
    naive_by[d] = {'hours':hours, 'person_hours':ph}
    naive_ph += ph

print(f"Naive total person-hours (this comparator): {naive_ph}")
print(f"Optimized total person-hours (from schedule): {opt_ph}")

# percent improvement (naive -> optimized)
if naive_ph > 0:
    pct_improve = 100.0 * (naive_ph - opt_ph) / naive_ph
else:
    pct_improve = None

summary = {
    'naive_person_hours': naive_ph,
    'optimized_person_hours': opt_ph,
    'percent_improvement_vs_naive': pct_improve
}

Path("outputs").mkdir(exist_ok=True)
with open(OUT_SUM, "w") as f:
    json.dump(summary, f, indent=2)

print("Summary:", summary)
print("Saved outputs/compare_naive_summary.json")