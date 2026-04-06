# scripts/estimate_outage_risk.py
"""
Estimate per-district outage risk (0-1) using available data:
- historical outages (recent counts)
- demand anomalies (predicted peak vs baseline)
- temperature extremes
- grid utilization (if available)
- population density (as an exposure multiplier)

Produces: outputs/estimated_outage_risk.csv with columns [district_id, outage_risk, risk_label]
"""
from pathlib import Path
import pandas as pd
import numpy as np

# File paths (adjust if needed)
DIST_REF = Path("data/geographic/ap_districts_reference.csv")
OUTAGES = Path("data/processed/outages.parquet")
MODEL = Path("data/processed/modeling_dataset.parquet")   # has predicted_peak or demand_mw
WEATHER = Path("data/processed/district_weather.parquet")
GRID = Path("data/processed/grid_infra.parquet")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Parameters (tunable)
RECENT_DAYS = 30
DEMAND_LOOKBACK_HOURS = 24 * 7  # use last week to compute baseline/avg
TEMP_EXTREME_C = 35.0  # temp threshold to count extreme hours
WEIGHTS = {
    "outage_count": 0.35,
    "demand_anomaly": 0.30,
    "temp_extreme": 0.15,
    "utilization": 0.10,
    "pop_density": 0.10
}

def safe_minmax(series):
    if series.isnull().all():
        return series.fillna(0.0)
    mn, mx = series.min(), series.max()
    if mn == mx:
        return series.apply(lambda _: 0.0)
    return (series - mn) / (mx - mn)

# Load district reference (population, pop_density)
if not DIST_REF.exists():
    raise SystemExit("Missing district reference: " + str(DIST_REF))
district_ref = pd.read_csv(DIST_REF)
district_ref['district_id'] = district_ref['district_id'].astype(str)

districts = district_ref['district_id'].unique().tolist()

# 1) Recent outage counts
outage_count = pd.Series(0, index=districts, dtype=float)
if OUTAGES.exists():
    outages = pd.read_parquet(OUTAGES)
    # find district column name
    possible = [c for c in outages.columns if 'district' in c.lower()]
    if possible:
        dcol = possible[0]
        outages[dcol] = outages[dcol].astype(str)
        # determine end time for recent window
        if 'end_timestamp' in outages.columns:
            outages['end_timestamp'] = pd.to_datetime(outages['end_timestamp'], errors='coerce')
            last_time = outages['end_timestamp'].max()
        elif 'timestamp' in outages.columns:
            outages['timestamp'] = pd.to_datetime(outages['timestamp'], errors='coerce')
            last_time = outages['timestamp'].max()
        else:
            last_time = None
        if last_time is None:
            # fallback: use entire dataset counts
            recent = outages.copy()
        else:
            cutoff = last_time - pd.Timedelta(days=RECENT_DAYS)
            if 'end_timestamp' in outages.columns:
                recent = outages[(outages['end_timestamp'] >= cutoff)]
            else:
                recent = outages[(outages['timestamp'] >= cutoff)]
        counts = recent.groupby(dcol).size().to_dict()
        for d, v in counts.items():
            if str(d) in outage_count.index:
                outage_count.loc[str(d)] = float(v)
# 2) Demand anomaly (use predicted_peak or demand_mw)
demand_anom = pd.Series(0.0, index=districts, dtype=float)
if MODEL.exists():
    md = pd.read_parquet(MODEL)
    # normalize column names
    if 'predicted_peak_24h' in md.columns:
        peak_col = 'predicted_peak_24h'
    elif 'predicted_peak_demand_mw' in md.columns:
        peak_col = 'predicted_peak_demand_mw'
    elif 'peak_demand_mw' in md.columns:
        peak_col = 'peak_demand_mw'
    elif 'demand_mw' in md.columns:
        peak_col = 'demand_mw'
    else:
        peak_col = None
    if peak_col:
        md['timestamp'] = pd.to_datetime(md['timestamp'], errors='coerce')
        # compute recent average vs historical baseline per district
        last_ts = md['timestamp'].max()
        recent_cut = last_ts - pd.Timedelta(hours=DEMAND_LOOKBACK_HOURS)
        baseline = md.groupby('district_id')[peak_col].median()
        recent_mean = md[md['timestamp'] >= recent_cut].groupby('district_id')[peak_col].mean()
        for d in districts:
            b = baseline.get(d, np.nan)
            r = recent_mean.get(d, np.nan)
            if pd.notna(b) and pd.notna(r):
                demand_anom.loc[d] = float(max(0.0, (r - b) / (b + 1e-9)))  # fractional positive anomaly
# 3) Temperature extremes
temp_extreme = pd.Series(0.0, index=districts, dtype=float)
if WEATHER.exists():
    wf = pd.read_parquet(WEATHER)
    if 'temperature' in wf.columns and 'timestamp' in wf.columns and 'district_id' in wf.columns:
        wf['timestamp'] = pd.to_datetime(wf['timestamp'], errors='coerce')
        last = wf['timestamp'].max()
        cutoff = last - pd.Timedelta(days=RECENT_DAYS)
        recent_w = wf[wf['timestamp'] >= cutoff]
        # fraction of hours over threshold
        frac = recent_w.groupby('district_id').apply(lambda g: (g['temperature'] > TEMP_EXTREME_C).mean() if len(g)>0 else 0.0)
        for d, v in frac.items():
            if str(d) in temp_extreme.index:
                temp_extreme.loc[str(d)] = float(v)
# 4) Grid utilization (if available)
utilization = pd.Series(0.0, index=districts, dtype=float)
if GRID.exists():
    gf = pd.read_parquet(GRID)
    # try common columns: district_id, utilization_ratio or capacity_mw, peak_load_mw
    if 'district_id' in gf.columns and 'utilization_ratio' in gf.columns:
        util = gf.groupby('district_id')['utilization_ratio'].mean()
        for d, v in util.items():
            if str(d) in utilization.index:
                utilization.loc[str(d)] = float(v)
    elif 'district' in gf.columns and 'peak_load_mw' in gf.columns and 'capacity_mw' in gf.columns:
        gf['district'] = gf['district'].astype(str)
        util = (gf['peak_load_mw'] / (gf['capacity_mw'].replace({0:np.nan}))).fillna(0)
        util = gf.groupby('district').apply(lambda g: (g['peak_load_mw'].sum() / max(1e-9, g['capacity_mw'].sum())))
        for d, v in util.items():
            if str(d) in utilization.index:
                utilization.loc[str(d)] = float(v)

# 5) Population density exposure (higher density -> more severe impact)
pop_density = pd.Series(0.0, index=districts, dtype=float)
if 'pop_density' in district_ref.columns:
    for _, r in district_ref.iterrows():
        d = str(r['district_id'])
        if d in pop_density.index:
            pop_density.loc[d] = float(r.get('pop_density', 0.0))
else:
    # fallback: if population and area present
    if 'population' in district_ref.columns and 'area_km2' in district_ref.columns:
        for _, r in district_ref.iterrows():
            d = str(r['district_id'])
            pop_density.loc[d] = float(r.get('population', 0.0)) / max(1.0, float(r.get('area_km2', 1.0)))

# Normalize components
o_norm = safe = lambda s: (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min())>0 else s*0
out_norm = safe_min = None
# we'll implement min-max safely
def minmax_safe(series):
    s = series.fillna(0.0).astype(float)
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

out_n = minmax_safe(outage_count)
d_n = minmax_safe(demand_anom)
t_n = minmax_safe(temp_extreme)
u_n = minmax_safe(utilization)
p_n = minmax_safe(pop_density)

# Composite risk
W = WEIGHTS
composite = W['outage_count']*out_n + W['demand_anomaly']*d_n + W['temp_extreme']*t_n + W['utilization']*u_n + W['pop_density']*p_n

# Normalize to 0-1
risk = minmax_safe(composite)

# Compact results
res = pd.DataFrame({
    'district_id': risk.index,
    'outage_risk': risk.values,
    'outage_count_30d': outage_count.values,
    'demand_anomaly': demand_anom.values,
    'temp_extreme_frac': temp_extreme.values,
    'utilization': utilization.values,
    'pop_density': pop_density.values
})

# risk label
def label(x):
    if x >= 0.66: return 'high'
    if x >= 0.33: return 'medium'
    return 'low'
res['risk_label'] = res['outage_risk'].apply(label)

# save
out_file = OUT / "estimated_outage_risk.csv"
res.to_csv(out_file, index=False)
print("Wrote", out_file)
print(res.sort_values('outage_risk', ascending=False).head(20).to_string(index=False))