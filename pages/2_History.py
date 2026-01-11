from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_app.components.feature_store_io import read_feature_group

# ---------------------------------------------------------------------
# Load local secrets (.env). Safe in Streamlit Cloud (ignored if missing)
# ---------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ---------------------------------------------------------------------
# Feature Group config (EDIT THESE IF NEEDED)
# ---------------------------------------------------------------------
ELECTRICITY_FG_NAME = "electricity"
ELECTRICITY_FG_VERSION = 1

WEATHER_FG_NAME = "weather"
WEATHER_FG_VERSION = 1

# Electricity schema
TS_COL = "date"   # <-- change if needed
PRICE_COL = "sek_per_kwh"   # <-- change if needed

# Weather schema (as you described)
WEATHER_TS_COL = "date"  # <-- change if your weather time column isn't "date"
CITY_COL = "city"        # <-- change if your city column has a different name

TEMP_COL = "temperature_2m"
WIND_COL = "wind_speed_10m"
PRECIP_COL = "precipitation"
CLOUD_COL = "cloud_cover"

CITY_VALUE = "stockholm"  # filter value

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Historical Data",
    page_icon="⚡",
    layout="wide",
)

st.title("📈 Historical Data")
st.caption("Electricity prices + weather features used by the model (from Hopsworks Feature Store).")

# ---------------------------------------------------------------------
# Load feature groups
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_electricity() -> pd.DataFrame:
    return read_feature_group(name=ELECTRICITY_FG_NAME, version=ELECTRICITY_FG_VERSION)

@st.cache_data(ttl=300)
def load_weather() -> pd.DataFrame:
    return read_feature_group(name=WEATHER_FG_NAME, version=WEATHER_FG_VERSION)

def compute_trailing_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    out = {
        "rows": int(len(s)),
        "mean_all": float(s.mean()) if len(s) else float("nan"),
        "std_all": float(s.std()) if len(s) > 1 else float("nan"),
    }
    windows = {"1m": 30, "6m": 182, "12m": 365}
    for k, n in windows.items():
        tail = s.tail(n)
        out[f"{k}_mean"] = float(tail.mean()) if len(tail) else float("nan")
        out[f"{k}_std"] = float(tail.std()) if len(tail) > 1 else float("nan")
    return out

try:
    df_el = load_electricity()
except Exception as e:
    st.error("Failed to load the electricity feature group from Hopsworks.")
    st.exception(e)
    st.stop()

# Developer-only schema hint
with st.expander("Developer: electricity schema (columns)"):
    st.write(list(df_el.columns))
    st.write("Row count:", len(df_el))

# Validate electricity schema
missing = []
if TS_COL not in df_el.columns:
    missing.append(f"Missing electricity time column `{TS_COL}`")
if PRICE_COL not in df_el.columns:
    missing.append(f"Missing electricity price column `{PRICE_COL}`")
if missing:
    st.error("Electricity schema mismatch. Update TS_COL / PRICE_COL:\n- " + "\n- ".join(missing))
    st.stop()

# Parse + sort electricity
df_el = df_el.copy()
df_el[TS_COL] = pd.to_datetime(df_el[TS_COL], utc=True, errors="coerce")
df_el = df_el.dropna(subset=[TS_COL]).sort_values(TS_COL)

# ---------------------------------------------------------------------
# Electricity stats / KPIs (full dataset)
# ---------------------------------------------------------------------
stats = compute_trailing_stats(df_el[PRICE_COL])

st.markdown("### ⚡ Electricity — Key statistics")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Mean (all)", f"{stats['mean_all']:.1f}")
c3.metric("Std dev (all)", f"{stats['std_all']:.1f}")

roll_df = pd.DataFrame(
    {
        "Window": ["Last 1 month", "Last 6 months", "Last 12 months"],
        "Mean": [stats["1m_mean"], stats["6m_mean"], stats["12m_mean"]],
        "Std dev": [stats["1m_std"], stats["6m_std"], stats["12m_std"]],
    }
)
roll_df["Mean"] = roll_df["Mean"].round(1)
roll_df["Std dev"] = roll_df["Std dev"].round(1)

st.dataframe(roll_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------
# Sidebar controls: timeframe + moving averages (electricity)
# ---------------------------------------------------------------------
st.sidebar.subheader("Electricity chart controls")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["Last 1M", "Last 3M", "Last 6M", "Last 12M", "YTD", "Last 3Y", "All", "Custom"],
    index=3,
)

show_mas = st.sidebar.checkbox("Show moving averages", value=True)
show_ma_30 = st.sidebar.checkbox("30d avg", value=True) if show_mas else False
show_ma_182 = st.sidebar.checkbox("6m avg", value=True) if show_mas else False
show_ma_365 = st.sidebar.checkbox("12m avg", value=False) if show_mas else False

min_ts = df_el[TS_COL].min()
max_ts = df_el[TS_COL].max()

def start_for_timeframe(tf: str) -> pd.Timestamp:
    end = max_ts
    if tf == "Last 1M":
        return end - pd.Timedelta(days=30)
    if tf == "Last 3M":
        return end - pd.Timedelta(days=90)
    if tf == "Last 6M":
        return end - pd.Timedelta(days=182)
    if tf == "Last 12M":
        return end - pd.Timedelta(days=365)
    if tf == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)
    if tf == "Last 3Y":
        return end - pd.Timedelta(days=365 * 3)
    if tf == "All":
        return min_ts
    return min_ts

if timeframe == "Custom":
    start_date, end_date = st.sidebar.date_input(
        "Custom date range",
        value=(min_ts.date(), max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_ts = start_for_timeframe(timeframe)
    end_ts = max_ts

df_el_plot = df_el[(df_el[TS_COL] >= start_ts) & (df_el[TS_COL] <= end_ts)].copy()
if df_el_plot.empty:
    st.warning("No electricity data in the selected timeframe.")
    st.stop()

# ---------------------------------------------------------------------
# Electricity plot
# ---------------------------------------------------------------------
st.subheader("📈 Historical electricity price")

fig_el = go.Figure()
fig_el.add_trace(go.Scatter(x=df_el_plot[TS_COL], y=df_el_plot[PRICE_COL], mode="lines", name="Price"))

if show_ma_30:
    df_el_plot["ma_30d"] = df_el_plot[PRICE_COL].rolling(30).mean()
    fig_el.add_trace(go.Scatter(x=df_el_plot[TS_COL], y=df_el_plot["ma_30d"], mode="lines", name="30d avg", line=dict(dash="dash")))
if show_ma_182:
    df_el_plot["ma_182d"] = df_el_plot[PRICE_COL].rolling(182).mean()
    fig_el.add_trace(go.Scatter(x=df_el_plot[TS_COL], y=df_el_plot["ma_182d"], mode="lines", name="6m avg", line=dict(dash="dot")))
if show_ma_365:
    df_el_plot["ma_365d"] = df_el_plot[PRICE_COL].rolling(365).mean()
    fig_el.add_trace(go.Scatter(x=df_el_plot[TS_COL], y=df_el_plot["ma_365d"], mode="lines", name="12m avg", line=dict(dash="dashdot")))

fig_el.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title="Price",
)

st.plotly_chart(fig_el, use_container_width=True)

st.subheader("📋 Latest electricity rows (filtered timeframe)")
st.dataframe(df_el_plot.tail(50), use_container_width=True)

# =====================================================================
# Weather section
# =====================================================================
st.markdown("---")
st.header("🌦 Weather — Stockholm")
st.caption("Weather feature group values used by the model (Stockholm).")

try:
    df_w = load_weather()
except Exception as e:
    st.error("Failed to load the weather feature group from Hopsworks.")
    st.exception(e)
    st.stop()

with st.expander("Developer: weather schema (columns)"):
    st.write(list(df_w.columns))
    st.write("Row count:", len(df_w))

# Validate weather schema
missing_w = []
for col in [WEATHER_TS_COL, CITY_COL, TEMP_COL, WIND_COL, PRECIP_COL, CLOUD_COL]:
    if col not in df_w.columns:
        missing_w.append(f"Missing weather column `{col}`")
if missing_w:
    st.error("Weather schema mismatch. Update WEATHER_TS_COL / CITY_COL / column names:\n- " + "\n- ".join(missing_w))
    st.stop()

df_w = df_w.copy()
df_w[WEATHER_TS_COL] = pd.to_datetime(df_w[WEATHER_TS_COL], utc=True, errors="coerce")
df_w = df_w.dropna(subset=[WEATHER_TS_COL]).sort_values(WEATHER_TS_COL)

# Filter Stockholm + align to same timeframe as electricity selection
df_w_city = df_w[df_w[CITY_COL].astype(str).str.lower() == CITY_VALUE.lower()].copy()
df_w_city = df_w_city[(df_w_city[WEATHER_TS_COL] >= start_ts) & (df_w_city[WEATHER_TS_COL] <= end_ts)].copy()

if df_w_city.empty:
    st.warning(f"No weather data found for {CITY_VALUE} in the selected timeframe.")
    st.stop()

# Weather summary KPIs (precip + cloud cover)
precip = pd.to_numeric(df_w_city[PRECIP_COL], errors="coerce")
cloud = pd.to_numeric(df_w_city[CLOUD_COL], errors="coerce")

wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("Rows", f"{len(df_w_city):,}")
wc2.metric("Precip mean", f"{precip.mean():.2f}" if precip.notna().any() else "—")
wc3.metric("Precip std", f"{precip.std():.2f}" if precip.notna().sum() > 1 else "—")
wc4.metric("Cloud mean", f"{cloud.mean():.1f}" if cloud.notna().any() else "—")

st.caption(
    f"Cloud cover range: "
    f"{cloud.min():.1f} – {cloud.max():.1f}" if cloud.notna().any() else "Cloud cover not available."
)

# Temperature plot
st.subheader("🌡 Temperature (2m) — Stockholm")
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(
    x=df_w_city[WEATHER_TS_COL],
    y=df_w_city[TEMP_COL],
    mode="lines",
    name="temperature_2m",
))
fig_temp.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title="Temperature (2m)",
)
st.plotly_chart(fig_temp, use_container_width=True)

# Wind plot
st.subheader("💨 Wind speed (10m) — Stockholm")
fig_wind = go.Figure()
fig_wind.add_trace(go.Scatter(
    x=df_w_city[WEATHER_TS_COL],
    y=df_w_city[WIND_COL],
    mode="lines",
    name="wind_speed_10m",
))
fig_wind.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title="Wind speed (10m)",
)
st.plotly_chart(fig_wind, use_container_width=True)

st.subheader("📋 Latest weather rows (Stockholm, filtered timeframe)")
st.dataframe(df_w_city.tail(50), use_container_width=True)
