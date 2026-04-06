# dashboard/app.py
"""
Streamlit dashboard for Smart Load Shedding Optimizer (Andhra Pradesh).
Run with: python -m streamlit run dashboard/app.py
"""
import os, sys
# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime

# optional folium map support
try:
    import folium
    from streamlit_folium import folium_static
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# project modules
from services.predictor import predict_snapshot_all, get_latest_snapshot
from optimizer.load_shedding_optimizer import calculate_impact_scores, allocate_load_shedding, generate_simple_schedule

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
DIST_REF = BASE_DIR / "data" / "geographic" / "ap_districts_reference.csv"
MODELING_DATA = BASE_DIR / "data" / "processed" / "modeling_dataset.parquet"
CRITICAL_INFRA = BASE_DIR / "data" / "processed" / "critical_infra.parquet"

st.set_page_config(layout="wide", page_title="Smart Load Shedding Optimizer — AP")
st.title("Smart Load Shedding Optimizer — Andhra Pradesh (POC)")
st.markdown("Interactive demo: forecasts, outage risk map, and optimized load-shedding schedule.")

# Sidebar controls
st.sidebar.header("Controls")
date_input = st.sidebar.date_input("Select reference date (for snapshot/visuals)",
                                   value=(datetime.date(2023, 5, 14)),
                                   min_value=datetime.date(2020, 5, 1),
                                   max_value=datetime.date(2023, 5, 31))
deficit = st.sidebar.slider("Power deficit to manage (MW)", 0, 5000, 1500, step=100)
run_button = st.sidebar.button("Run prediction & optimize")

@st.cache_data
def load_district_ref():
    if DIST_REF.exists():
        return pd.read_csv(DIST_REF)
    return pd.DataFrame()

@st.cache_data
def load_outputs():
    out = {}
    out['forecast'] = pd.read_csv(OUT_DIR / "forecast.csv", parse_dates=["timestamp"]) if (OUT_DIR / "forecast.csv").exists() else pd.DataFrame()
    out['forecast_24h'] = pd.read_csv(OUT_DIR / "forecast_24h.csv", parse_dates=["timestamp"]) if (OUT_DIR / "forecast_24h.csv").exists() else pd.DataFrame()
    out['risk'] = pd.read_csv(OUT_DIR / "outage_risk.csv", parse_dates=["timestamp"]) if (OUT_DIR / "outage_risk.csv").exists() else pd.DataFrame()
    out['alloc'] = pd.read_csv(OUT_DIR / "load_shedding_allocations.csv") if (OUT_DIR / "load_shedding_allocations.csv").exists() else pd.DataFrame()
    out['schedule'] = pd.read_csv(OUT_DIR / "load_shedding_schedule.csv") if (OUT_DIR / "load_shedding_schedule.csv").exists() else pd.DataFrame()
    return out

def run_pipeline_and_load(deficit_val):
    forecasts, risks = predict_snapshot_all()
    # prepare input for optimizer (safe merge)
    ref = load_district_ref()
    df = forecasts.merge(risks[['district_id', 'outage_risk']], on='district_id', how='left')
    if 'predicted_peak_demand_mw' in df.columns:
        df = df.rename(columns={'predicted_peak_demand_mw':'peak_demand_mw'})
    # safe merge only with available columns
    if not ref.empty:
        possible = ['district_id', 'pop_density', 'district_name', 'latitude', 'longitude']
        avail = [c for c in possible if c in ref.columns]
        if 'district_id' in avail:
            df = df.merge(ref[avail], on='district_id', how='left')
    df['peak_demand_mw'] = pd.to_numeric(df.get('peak_demand_mw', 0), errors='coerce').fillna(0)
    df['outage_risk'] = pd.to_numeric(df.get('outage_risk', 0), errors='coerce').fillna(0.0)
    scored = calculate_impact_scores( df, critical_infra=pd.read_parquet(CRITICAL_INFRA) if CRITICAL_INFRA.exists() else pd.DataFrame() )
    allocations, summary = allocate_load_shedding(scored, float(deficit_val))
    schedule_df, hourly_totals = generate_simple_schedule(allocations)
    # save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(OUT_DIR / "forecast.csv", index=False)
    risks.to_csv(OUT_DIR / "outage_risk.csv", index=False)
    allocations.to_csv(OUT_DIR / "load_shedding_allocations.csv", index=False)
    schedule_df.to_csv(OUT_DIR / "load_shedding_schedule.csv", index=False)
    return load_outputs()

# If Run pressed -> recompute (and overwrite outputs)
# Demo mode: always load precomputed outputs
outputs = load_outputs()

if run_button:
    st.info("This deployed demo uses precomputed outputs. Run the full pipeline locally to refresh results, then push updated outputs to GitHub.")

district_ref = load_district_ref()
forecasts = outputs['forecast']
forecasts_24h = outputs['forecast_24h']
risks = outputs['risk']
allocations = outputs['alloc']
schedule_df = outputs['schedule']

tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Outage Risk Map", "Allocations & Schedule"])

with tab1:
    st.header("Demand forecast (Plotly)")
    if forecasts.empty and forecasts_24h.empty:
        st.info("No forecast data found. Press 'Run prediction & optimize' or generate a 24h forecast.")
    else:
        # use forecasts_24h if available for time-series richness, else forecasts
        source = forecasts_24h if not forecasts_24h.empty else forecasts
        all_districts = sorted(source['district_id'].unique())
        default = all_districts[:6] if len(all_districts) >= 6 else all_districts
        sel = st.multiselect("Districts", all_districts, default=default)
        plot_df = source[source['district_id'].isin(sel)].copy()
        if not plot_df.empty and plot_df['timestamp'].nunique() > 1:
            fig = px.line(plot_df, x='timestamp', y='predicted_peak_demand_mw' if 'predicted_peak_demand_mw' in plot_df.columns else 'peak_demand_mw',
                          color='district_id', labels={'value':'Predicted Peak Demand (MW)'}, markers=True)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No multi-timestamp forecast available; consider running generate_24hr_forecast script or the pipeline.")
        st.subheader("Forecast vs Actual (single district)")
        district_for_compare = st.selectbox("Select district for forecast vs actual", sel or default)
        if district_for_compare:
            if MODELING_DATA.exists():
                md = pd.read_parquet(MODELING_DATA)
                md['timestamp'] = pd.to_datetime(md['timestamp'], errors='coerce')
                actual = md[md['district_id'] == district_for_compare].sort_values('timestamp')
                # try multi-timestamp forecasts first: prefer 24h file, then main forecasts
                fc = pd.DataFrame()
                if not forecasts_24h.empty:
                    fc = forecasts_24h[forecasts_24h['district_id']==district_for_compare].sort_values('timestamp')
                if fc.empty and not forecasts.empty:
                    fc = forecasts[forecasts['district_id']==district_for_compare].sort_values('timestamp')
                show_message = False
                if fc.empty:
                    # attempt snapshot fallback
                    snap = get_latest_snapshot()
                    fc = snap[snap['district_id']==district_for_compare].sort_values('timestamp')
                    show_message = True
                fig2 = px.line()
                if not actual.empty:
                    if 'peak_demand_mw' in actual.columns:
                        fig2.add_scatter(x=actual['timestamp'], y=actual['peak_demand_mw'], mode='lines', name='Actual Peak (MW)')
                    elif 'demand_mw' in actual.columns:
                        fig2.add_scatter(x=actual['timestamp'], y=actual['demand_mw'], mode='lines', name='Actual (MW)')
                if not fc.empty:
                    pred_col = 'predicted_peak_demand_mw' if 'predicted_peak_demand_mw' in fc.columns else ('peak_demand_mw' if 'peak_demand_mw' in fc.columns else fc.columns[2])
                    fig2.add_scatter(x=fc['timestamp'], y=fc[pred_col], mode='markers+lines' if len(fc)>1 else 'markers', name='Predicted')
                if fig2.data:
                    fig2.update_layout(height=450, xaxis_title='Timestamp', yaxis_title='MW')
                    st.plotly_chart(fig2, use_container_width=True)
                    if show_message:
                        st.info("Predicted series is a single snapshot. Run scripts/generate_24hr_forecast.py to create a 24-hour forecast for a line.")
                else:
                    st.info("Not enough data for forecast vs actual for this district.")
            else:
                st.info("Modeling dataset not found for actuals.")

with tab2:
    st.header("Outage Risk Map")
    if risks.empty:
        st.info("No outage risk data available. Run pipeline to compute.")
    else:
        map_df = risks.merge(district_ref, on='district_id', how='left') if not district_ref.empty else risks.copy()
        # if folium available and lat/lon exist show map
        if HAS_FOLIUM and 'latitude' in map_df.columns and 'longitude' in map_df.columns:
            center_lat = map_df['latitude'].mean()
            center_lon = map_df['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            for _, r in map_df.iterrows():
                lat = r.get('latitude') or center_lat
                lon = r.get('longitude') or center_lon
                risk = float(r.get('outage_risk', 0) or 0)
                color = 'red' if risk > 0.66 else ('orange' if risk > 0.33 else 'green')
                folium.CircleMarker([lat, lon], radius=8, color=color, fill=True, fill_opacity=0.7,
                                    popup=f"{r['district_id']} - {r.get('district_name','')}: risk={risk:.2f}").add_to(m)
            folium_static(m, width=800)
        else:
            # fallback to bar chart
            bar_df = map_df.sort_values('outage_risk', ascending=False)
            fig = px.bar(bar_df, x='district_id', y='outage_risk', labels={'outage_risk':'Outage Risk (0-1)'})
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Allocations & Schedule")
    if allocations.empty:
        st.info("No allocations to display. Run pipeline to compute.")
    else:
        st.subheader("Allocations (per-district)")
        st.dataframe(allocations.sort_values('shed_mw', ascending=False).reset_index(drop=True))
        csv_alloc = allocations.to_csv(index=False).encode('utf-8')
        st.download_button("Download allocations CSV", csv_alloc, file_name="load_shedding_allocations.csv")
    st.subheader("Schedule (binary per-hour)")
    if schedule_df.empty:
        st.info("No schedule available.")
    else:
        st.dataframe(schedule_df.head(50))
        csv_sch = schedule_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download schedule CSV", csv_sch, file_name="load_shedding_schedule.csv")

st.caption("Change date and deficit in the sidebar and press 'Run prediction & optimize' to update outputs.")