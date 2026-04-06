# dashboard/app.py
"""
Streamlit dashboard for Smart Load Shedding Optimizer (Andhra Pradesh).
Demo mode: reads precomputed outputs from the repo.
Run with: python -m streamlit run dashboard/app.py
"""

import os
import sys
import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# optional folium map support
try:
    import folium
    from streamlit_folium import folium_static
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
DIST_REF = BASE_DIR / "data" / "geographic" / "ap_districts_reference.csv"
MODELING_DATA = BASE_DIR / "data" / "processed" / "modeling_dataset.parquet"

st.set_page_config(layout="wide", page_title="Smart Load Shedding Optimizer — AP")
st.title("Smart Load Shedding Optimizer — Andhra Pradesh (POC)")
st.markdown("Interactive demo: forecasts, outage risk map, and optimized load-shedding schedule.")

st.sidebar.header("Controls")
date_input = st.sidebar.date_input(
    "Select reference date (for visuals)",
    value=datetime.date(2023, 5, 14),
    min_value=datetime.date(2020, 5, 1),
    max_value=datetime.date(2023, 5, 31),
)
deficit = st.sidebar.slider("Power deficit to manage (MW)", 0, 5000, 1500, step=100)
run_button = st.sidebar.button("Run prediction & optimize")


@st.cache_data
def load_district_ref():
    if DIST_REF.exists():
        return pd.read_csv(DIST_REF)
    return pd.DataFrame()


@st.cache_data
def load_outputs():
    outputs = {}

    try:
        outputs["forecast_24h"] = pd.read_csv(OUT_DIR / "forecast_24h.csv", parse_dates=["timestamp"])
    except Exception:
        outputs["forecast_24h"] = pd.DataFrame()

    try:
        outputs["forecast"] = pd.read_csv(OUT_DIR / "forecast.csv", parse_dates=["timestamp"])
    except Exception:
        outputs["forecast"] = pd.DataFrame()

    try:
        outputs["risk"] = pd.read_csv(OUT_DIR / "outage_risk.csv", parse_dates=["timestamp"])
    except Exception:
        outputs["risk"] = pd.DataFrame()

    try:
        outputs["alloc"] = pd.read_csv(OUT_DIR / "load_shedding_allocations.csv")
    except Exception:
        outputs["alloc"] = pd.DataFrame()

    try:
        outputs["schedule"] = pd.read_csv(OUT_DIR / "load_shedding_schedule.csv")
    except Exception:
        outputs["schedule"] = pd.DataFrame()

    return outputs


# Demo mode: always load precomputed outputs
outputs = load_outputs()

if run_button:
    st.info("This deployed demo uses precomputed outputs. Run the full pipeline locally, then push updated outputs to GitHub.")

district_ref = load_district_ref()
forecasts = outputs.get("forecast", pd.DataFrame())
forecasts_24h = outputs.get("forecast_24h", pd.DataFrame())
risks = outputs.get("risk", pd.DataFrame())
allocations = outputs.get("alloc", pd.DataFrame())
schedule_df = outputs.get("schedule", pd.DataFrame())

tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Outage Risk Map", "Allocations & Schedule"])

with tab1:
    st.header("Demand forecast")
    if forecasts.empty and forecasts_24h.empty:
        st.info("No forecast data found.")
    else:
        source = forecasts_24h if not forecasts_24h.empty else forecasts
        all_districts = sorted(source["district_id"].unique())
        default = all_districts[:6] if len(all_districts) >= 6 else all_districts
        sel = st.multiselect("Districts", all_districts, default=default)
        plot_df = source[source["district_id"].isin(sel)].copy()

        if not plot_df.empty and plot_df["timestamp"].nunique() > 1:
            y_col = "predicted_peak_demand_mw" if "predicted_peak_demand_mw" in plot_df.columns else "peak_demand_mw"
            fig = px.line(
                plot_df,
                x="timestamp",
                y=y_col,
                color="district_id",
                markers=True,
                labels={y_col: "Predicted Peak Demand (MW)"},
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No multi-timestamp forecast available.")

with tab2:
    st.header("Outage Risk Map")
    if risks.empty:
        st.info("No outage risk data available.")
    else:
        map_df = risks.merge(district_ref, on="district_id", how="left") if not district_ref.empty else risks.copy()

        if HAS_FOLIUM and "latitude" in map_df.columns and "longitude" in map_df.columns:
            center_lat = map_df["latitude"].mean()
            center_lon = map_df["longitude"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

            for _, r in map_df.iterrows():
                lat = r.get("latitude") or center_lat
                lon = r.get("longitude") or center_lon
                risk = float(r.get("outage_risk", 0) or 0)
                color = "red" if risk > 0.66 else ("orange" if risk > 0.33 else "green")
                folium.CircleMarker(
                    [lat, lon],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{r['district_id']} - {r.get('district_name', '')}: risk={risk:.2f}",
                ).add_to(m)

            folium_static(m, width=800)
        else:
            bar_df = map_df.sort_values("outage_risk", ascending=False)
            fig = px.bar(bar_df, x="district_id", y="outage_risk", labels={"outage_risk": "Outage Risk (0-1)"})
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Allocations & Schedule")

    if allocations.empty:
        st.info("No allocations to display.")
    else:
        st.subheader("Allocations (per district)")
        st.dataframe(allocations.sort_values("shed_mw", ascending=False).reset_index(drop=True))
        st.download_button(
            "Download allocations CSV",
            allocations.to_csv(index=False).encode("utf-8"),
            file_name="load_shedding_allocations.csv",
        )

    st.subheader("Schedule")
    if schedule_df.empty:
        st.info("No schedule available.")
    else:
        st.dataframe(schedule_df.head(50))
        st.download_button(
            "Download schedule CSV",
            schedule_df.to_csv(index=False).encode("utf-8"),
            file_name="load_shedding_schedule.csv",
        )

st.caption("Deployed demo uses precomputed outputs stored in the repository.")