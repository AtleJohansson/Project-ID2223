from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components.feature_store_io import read_feature_group

# Load local secrets (.env). Safe in Streamlit Cloud (ignored if missing)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ---------------------------------------------------------------------
# Config (EDIT THESE ONCE)
# ---------------------------------------------------------------------
ELECTRICITY_FG_NAME = "electricity"
ELECTRICITY_FG_VERSION = 1

TS_COL = "date"          # <-- change if needed
PRICE_COL = "sek_per_kwh"          # <-- change if needed
PRED_COL = "sek_per_kwh" # <-- change if your prediction column is named differently

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Sidebar title
st.sidebar.title("Electricity Dashboard")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_electricity() -> pd.DataFrame:
    return read_feature_group(name=ELECTRICITY_FG_NAME, version=ELECTRICITY_FG_VERSION)

try:
    df = load_electricity()
except Exception as e:
    st.error("Failed to load the electricity feature group from Hopsworks.")
    st.exception(e)
    st.stop()

# Validate schema
missing = []
for col in [TS_COL, PRICE_COL]:
    if col not in df.columns:
        missing.append(col)

if missing:
    st.error(f"Missing required columns in `{ELECTRICITY_FG_NAME}`: {missing}")
    with st.expander("Developer: available columns"):
        st.write(list(df.columns))
    st.stop()

# Parse + sort
df = df.copy()
df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True, errors="coerce")
df = df.dropna(subset=[TS_COL]).sort_values(TS_COL)

# Optional: if predictions column doesn't exist, still show history
has_pred = PRED_COL in df.columns

# ---------------------------------------------------------------------
# Landing content
# ---------------------------------------------------------------------
st.title("⚡ Electricity Price Forecasting")
st.markdown(
    """
This dashboard shows:
- **Historical electricity prices**
- **Model forecasts** produced from weather + market features (Hopsworks Feature Store)

Use the pages in the sidebar to explore the full history, weather features, and detailed diagnostics.
"""
)

st.markdown("---")

# ---------------------------------------------------------------------
# Simple chart: price vs prediction
# ---------------------------------------------------------------------
st.subheader("📈 Electricity price vs prediction (recent window)")

# Show last N points on landing for clarity
N = 1000
df_plot = df.tail(N).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_plot[TS_COL],
    y=df_plot[PRICE_COL],
    mode="lines",
    name="Actual price",
))

if has_pred:
    # ensure numeric-ish
    df_plot[PRED_COL] = pd.to_numeric(df_plot[PRED_COL], errors="coerce")
    fig.add_trace(go.Scatter(
        x=df_plot[TS_COL],
        y=df_plot[PRED_COL],
        mode="lines",
        name="Prediction",
        line=dict(dash="dash"),
    ))
else:
    st.info(f"No prediction column `{PRED_COL}` found in `{ELECTRICITY_FG_NAME}`. Showing only actual prices.")

fig.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title="Price",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Small table
# ---------------------------------------------------------------------
st.subheader("📋 Latest rows")
cols_to_show = [TS_COL, PRICE_COL] + ([PRED_COL] if has_pred else [])
st.dataframe(df[cols_to_show].tail(30), use_container_width=True)
