# scripts/generate_node_prices.py
"""
Generate synthetic hourly node/hub prices for May 1, 2020 - May 31, 2023.
If data/processed/modeling_dataset.parquet exists with predicted_peak_24h by district,
the script will aggregate that signal to a hub demand proxy and create correlated prices.
Otherwise it generates seasonal synthetic demand.

Outputs:
 - outputs/node_hub_prices_full.csv with columns [timestamp,node_id,day_ahead_price,intraday_price]
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
start = pd.Timestamp("2020-05-01 00:00:00")
end = pd.Timestamp("2023-05-31 23:00:00")
rng = pd.date_range(start, end, freq="H")
hub_id = "AP-HUB"
node_ids = ["AP-01","AP-02","AP-03","AP-04","AP-05"]  # example nodes

# Load SLS modeling dataset if available (to correlate)
md_path = Path("data/processed/modeling_dataset.parquet")
if md_path.exists():
    try:
        md = pd.read_parquet(md_path)
        # aggregate predicted_peak_24h to hourly hub demand proxy if column exists
        if 'predicted_peak_24h' in md.columns and 'timestamp' in md.columns and 'district_id' in md.columns:
            md['timestamp'] = pd.to_datetime(md['timestamp'], errors='coerce')
            agg = md.groupby('timestamp')['predicted_peak_24h'].sum().reindex(rng).interpolate().fillna(method='ffill').fillna(method='bfill')
            demand_series = agg.values
        else:
            demand_series = None
    except Exception:
        demand_series = None
else:
    demand_series = None

# If no SLS demand series, create synthetic seasonality + noise
if demand_series is None:
    # daily + seasonal + trend
    hours = len(rng)
    day_of_year = rng.dayofyear.values
    daily = 1.0 + 0.3*np.sin(2*np.pi*(rng.hour.values)/24.0)      # daily cycle
    annual = 1.0 + 0.2*np.sin(2*np.pi*(day_of_year)/365.0)       # seasonal
    trend = 1.0 + 0.05 * ((rng.year - 2020) + rng.dayofyear/365.0)
    noise = np.random.normal(0, 0.05, size=hours)
    base_demand = 1000.0  # arbitrary base units
    demand_series = base_demand * daily * annual * trend * (1 + noise)

# Normalize demand proxy
demand_norm = (demand_series - np.mean(demand_series)) / (np.std(demand_series) + 1e-9)

# Price generation function: price = base + alpha*demand_norm + outage_shock + noise
hub_base = 2.5  # base day-ahead price in INR per unit (synthetic)
alpha = 0.5     # sensitivity to demand proxy
np.random.seed(42)
outages_path = Path("data/processed/outages.parquet")
outage_shock = np.zeros(len(rng))
if outages_path.exists():
    try:
        out_df = pd.read_parquet(outages_path)
        # expand outage events to hourly flags for any district; treat flag as shock multiplier
        if 'start_timestamp' in out_df.columns and 'end_timestamp' in out_df.columns:
            for _, r in out_df.iterrows():
                try:
                    st = pd.to_datetime(r['start_timestamp'], errors='coerce')
                    ed = pd.to_datetime(r['end_timestamp'], errors='coerce')
                    if pd.isna(st) or pd.isna(ed):
                        continue
                    idx = ((rng >= st) & (rng <= ed))
                    outage_shock[idx] += 0.5  # add fixed shock 0.5 per outage hour
                except Exception:
                    continue
    except Exception:
        outage_shock = np.zeros(len(rng))

# Build DataFrame rows
rows = []
for i, ts in enumerate(rng):
    hub_price_day = hub_base + alpha * demand_norm[i] + outage_shock[i] + np.random.normal(0, 0.03)
    hub_price_intraday = hub_price_day + np.random.normal(0, 0.02)
    rows.append((ts, hub_id, round(float(hub_price_day),4), round(float(hub_price_intraday),4)))
    # node prices: add node-specific idiosyncratic component
    for j, nid in enumerate(node_ids):
        node_dispersion = (j - len(node_ids)/2) * 0.05  # small node-specific offset
        node_price_day = hub_price_day + node_dispersion + np.random.normal(0,0.02)
        node_price_intraday = node_price_day + np.random.normal(0,0.02)
        rows.append((ts, nid, round(float(node_price_day),4), round(float(node_price_intraday),4)))

df = pd.DataFrame(rows, columns=["timestamp","node_id","day_ahead_price","intraday_price"])
# Save full CSV (may be large)
out_file = OUT / "node_hub_prices_full.csv"
df.to_csv(out_file, index=False)
print("Wrote", out_file, "shape:", df.shape)