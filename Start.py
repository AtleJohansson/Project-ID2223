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
ELECTRICITY_FG_NAME = "electricity_hourly"
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
st.sidebar.title("")

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
This dashboard presents the results of a serverless machine learning project developed by **Atle Johansson** and
**Robert Bromans** as part of the course **ID2223 – Scalable Machine Learning and Deep Learning** at **KTH Royal Institute of Technology**.

The project demonstrates an end-to-end, production-oriented machine learning pipeline built on a modern cloud-based
architecture. **Hopsworks** is used as the backend feature store and model infrastructure, while **Streamlit** serves
as the interactive frontend for data exploration, visualization, and result presentation.

Historical electricity price data is combined with weather information from multiple regions and cities across Sweden.
These datasets are stored and managed in the Hopsworks Feature Store and used to train an **XGBoost regression model**
designed to capture temporal patterns and weather-driven effects in electricity prices.

Once trained, the model generates forecasts of future electricity prices, which are visualized alongside historical
observations in this dashboard. The application allows users to explore recent price trends, compare predictions to
actual values, and gain insight into the data and features used by the model.

Use the pages in the sidebar to explore historical data, weather features, and detailed diagnostics behind the
electricity price forecasts.
"""
)

st.subheader("📈 Electricity price vs prediction (last 1000 points)")
st.markdown(
    "The chart below shows the result of this project, visit the other pages for more granular data and explanations."
    
)

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
