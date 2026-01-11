from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components.feature_store_io import read_feature_group

# ---------------------------------------------------------------------
# Load local secrets (.env). Safe in Streamlit Cloud (ignored if missing)
# ---------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------
ELECTRICITY_FG_NAME = "electricity_hourly"
ELECTRICITY_FG_VERSION = 1

PREDICTIONS_FG_NAME = "price_predictions"
PREDICTIONS_FG_VERSION = 1  # change if needed

# ---------------------------------------------------------------------
# Schema (edit if needed)
# ---------------------------------------------------------------------
TS_COL = "date"
PRICE_COL = "sek_per_kwh"
PRED_COL = "predicted_sek_per_kwh"

# ---------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------
UNITS = {"price": "SEK/kWh"}

def axis_title(name: str, unit: str | None = None) -> str:
    return f"{name} ({unit})" if unit else name

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.sidebar.title("")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_electricity() -> pd.DataFrame:
    return read_feature_group(
        name=ELECTRICITY_FG_NAME,
        version=ELECTRICITY_FG_VERSION,
    )

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return read_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
    )

try:
    df_hist = load_electricity()
except Exception as e:
    st.error("Failed to load electricity feature group from Hopsworks.")
    st.exception(e)
    st.stop()

try:
    df_pred = load_predictions()
except Exception as e:
    st.error("Failed to load price_predictions feature group from Hopsworks.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------------
# Validate schemas
# ---------------------------------------------------------------------
missing_hist = [c for c in [TS_COL, PRICE_COL] if c not in df_hist.columns]
if missing_hist:
    st.error(f"Missing required columns in `{ELECTRICITY_FG_NAME}`: {missing_hist}")
    with st.expander("Developer: electricity_hourly columns"):
        st.write(list(df_hist.columns))
    st.stop()

missing_pred = [c for c in [TS_COL, PRED_COL] if c not in df_pred.columns]
if missing_pred:
    st.error(f"Missing required columns in `{PREDICTIONS_FG_NAME}`: {missing_pred}")
    with st.expander("Developer: price_predictions columns"):
        st.write(list(df_pred.columns))
    st.stop()

# ---------------------------------------------------------------------
# Parse + sort
# ---------------------------------------------------------------------
df_hist = df_hist.copy()
df_hist[TS_COL] = pd.to_datetime(df_hist[TS_COL], utc=True, errors="coerce")
df_hist = df_hist.dropna(subset=[TS_COL]).sort_values(TS_COL)
df_hist[PRICE_COL] = pd.to_numeric(df_hist[PRICE_COL], errors="coerce")

df_pred = df_pred.copy()
df_pred[TS_COL] = pd.to_datetime(df_pred[TS_COL], utc=True, errors="coerce")
df_pred = df_pred.dropna(subset=[TS_COL]).sort_values(TS_COL)
df_pred[PRED_COL] = pd.to_numeric(df_pred[PRED_COL], errors="coerce")

# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------
st.title("⚡ Electricity Price Forecasting")
st.markdown(
    """
This page shows the most recent historical electricity prices together with the latest
model-generated forecasts. The historical data comes from the electricity feature group,
while the forecast is produced by an XGBoost model and stored in the `price_predictions`
feature group.

The chart below displays the last 20 days of historical prices and the corresponding
future price predictions.
"""
)

# ---------------------------------------------------------------------
# Plot window: last 20 days (hourly → 480 points)
# ---------------------------------------------------------------------
st.subheader("📈 Electricity price — history vs forecast")
st.caption(f"Unit: {UNITS['price']} • Timestamps shown in UTC")

N = 24 * 20  # 480 hours
df_hist_plot = df_hist.tail(N).copy()
if df_hist_plot.empty:
    st.warning("No historical electricity data available.")
    st.stop()

# Extend x-axis to include future predictions
x_start = df_hist_plot[TS_COL].min()
x_end = max(df_hist_plot[TS_COL].max(), df_pred[TS_COL].max())

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig = go.Figure()

# Historical prices (BLUE)
fig.add_trace(
    go.Scatter(
        x=df_hist_plot[TS_COL],
        y=df_hist_plot[PRICE_COL],
        mode="lines",
        name="Actual price",
        line=dict(color="#1f77b4", width=3),  # blue
    )
)

# Forecast prices (GREEN)
if not df_pred.empty:
    fig.add_trace(
        go.Scatter(
            x=df_pred[TS_COL],
            y=df_pred[PRED_COL],
            mode="lines",
            name="Forecast",
            line=dict(color="#2ca02c", width=3),  # green
        )
    )
else:
    st.info("No forecast data available to plot.")

fig.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title=axis_title("Price", UNITS["price"]),
)

fig.update_xaxes(range=[x_start, x_end])

st.plotly_chart(fig, use_container_width=True)
