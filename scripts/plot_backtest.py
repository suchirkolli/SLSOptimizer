# scripts/plot_backtest.py
"""
Simple backtest equity curve using hub price as proxy.
Writes outputs/fig_backtest_equity_curve.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
MD = Path("data/processed/modeling_dataset.parquet")
PR = Path("outputs/node_hub_prices_full.csv")

if not MD.exists():
    sys.exit("Missing modeling dataset")
md = pd.read_parquet(MD)
md['timestamp'] = pd.to_datetime(md['timestamp'], errors='coerce')

# state signal z
for c in ['predicted_peak_24h','predicted_peak_demand_mw','peak_demand_mw','demand_mw']:
    if c in md.columns:
        peak_col = c; break
else:
    sys.exit("No predicted peak column")

state = md.groupby('timestamp')[peak_col].sum().rename('state_peak').to_frame()
state['z'] = (state['state_peak'] - state['state_peak'].rolling(24*90,min_periods=1).mean()) / state['state_peak'].rolling(24*90,min_periods=1).std().replace(0,1)

# load hub price or synth
if PR.exists():
    pr = pd.read_csv(PR, parse_dates=['timestamp'])
    hub = 'AP-HUB' if 'AP-HUB' in pr['node_id'].unique() else pr['node_id'].mode().iloc[0]
    hubp = pr[pr['node_id']==hub].set_index('timestamp').sort_index()
    price = hubp['intraday_price'].rename('price')
    state = state.join(price, how='left')
    state['price'] = state['price'].interpolate().fillna(method='ffill').fillna(method='bfill')
else:
    arr = state['state_peak'].values
    arrn = (arr - arr.mean())/(arr.std()+1e-9)
    state['price'] = 2.5 + 0.4*arrn + np.random.normal(0,0.03,len(arrn))

state = state.sort_index().dropna(subset=['price'])
# daily returns proxy
state['ret'] = state['price'].pct_change().fillna(0)

# signal: entry where z>1 (enter next hour) hold for 24 hours
zth = 1.0
pos = pd.Series(0, index=state.index, dtype=float)
entries = state['z'] > zth
for ts in state.index[entries]:
    entry = ts + pd.Timedelta(hours=1)
    exit = ts + pd.Timedelta(hours=24)
    pos.loc[entry:exit] += 1
# limit position to 1 max
pos = pos.clip(upper=1)

pnl = pos * state['ret']
cum = pnl.cumsum().fillna(method='ffill')

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode='lines', name='Cumulative PnL'))
fig.update_layout(title='Backtest: simple state-z strategy (proxy price)', xaxis_title='Time', yaxis_title='Cumulative PnL')
fig.write_image(str(OUT/'fig_backtest_equity_curve.png'), width=1000, height=600)
print("Saved", OUT/'fig_backtest_equity_curve.png')