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
# Feature Group config (EDIT THESE IF NEEDED)
# ---------------------------------------------------------------------
ELECTRICITY_FG_NAME = "electricity_hourly"
ELECTRICITY_FG_VERSION = 1

WEATHER_FG_NAME = "weather"
WEATHER_FG_VERSION = 1

# Electricity schema
TS_COL = "date"            # <-- change if needed
PRICE_COL = "sek_per_kwh"  # <-- change if needed

# Weather schema (NEW: city-specific columns)
WEATHER_TS_COL = "date"    # <-- change if your weather time column isn't "date"

# City toggle options (must match your column suffixes)
CITY_OPTIONS = {
    "Stockholm": "stockholm",
    "Uppsala": "uppsala",
    "Västerås": "vasteras",
    "Örebro": "orebro",
    "Karlstad": "karlstad",
    "Sundsvall": "sundsvall",
    "Malmö": "malmo",
}
DEFAULT_CITY_LABEL = "Stockholm"

# Column name templates
TEMP_TPL = "temperature_2m_{city}"
WIND_TPL = "wind_speed_10m_{city}"
PRECIP_TPL = "precipitation_{city}"
CLOUD_TPL = "cloud_cover_{city}"

# ---------------------------------------------------------------------
# Units + formatting (EDIT IF NEEDED)
# ---------------------------------------------------------------------
UNITS = {
    "price": "SEK/kWh",
    "temp": "°C",
    "wind": "m/s",   # set to "km/h" if your feature group stores that
    "precip": "mm",  # precipitation typically mm per time step (hour/day)
    "cloud": "%",    # cloud cover commonly 0–100 (%), sometimes 0–1 (handled below)
}

DECIMALS = {
    "price": 3,
    "temp": 1,
    "wind": 1,
    "precip": 2,
    "cloud": 1,
}

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------
LINE_WIDTH = 3  # All chart lines solid, width 3

def fmt(x, decimals: int = 2, unit: str | None = None, na: str = "—") -> str:
    """Safe number formatter with optional unit suffix."""
    try:
        if x is None:
            return na
        if isinstance(x, float) and pd.isna(x):
            return na
        if isinstance(x, (pd.Timestamp,)):
            return str(x)
        v = float(x)
        s = f"{v:,.{decimals}f}"
        return f"{s} {unit}" if unit else s
    except Exception:
        return na

def axis_title(name: str, unit: str | None = None) -> str:
    return f"{name} ({unit})" if unit else name

def is_prob_0_1(s: pd.Series) -> bool:
    """Heuristic: detect if values look like 0..1 proportions."""
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return False
    return (s2.min() >= 0) and (s2.max() <= 1.2)

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

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

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Historical Data",
    page_icon="⚡",
    layout="wide",
)

st.title("📈 Historical Data")
st.markdown(
    """
This page shows the historical electricity prices and weather conditions used to train the prediction model.

The electricity price time series represents past market prices, while the weather data (temperature, wind speed,
precipitation, and cloud cover for selected cities in Sweden) captures external factors that influence electricity
supply and demand.

Together, these features are used to train an XGBoost regression model that learns historical patterns and
relationships in the data. Once trained, the model uses the most recent observations to predict future electricity prices.

All values shown here are retrieved from the Hopsworks Feature Store and displayed with their corresponding units
to ensure clarity and transparency.
"""
)

# ---------------------------------------------------------------------
# Load feature groups
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_electricity() -> pd.DataFrame:
    return read_feature_group(name=ELECTRICITY_FG_NAME, version=ELECTRICITY_FG_VERSION)

@st.cache_data(ttl=300)
def load_weather() -> pd.DataFrame:
    return read_feature_group(name=WEATHER_FG_NAME, version=WEATHER_FG_VERSION)

# ---------------------------------------------------------------------
# Electricity
# ---------------------------------------------------------------------
try:
    df_el = load_electricity()
except Exception as e:
    st.error("Failed to load the electricity feature group from Hopsworks.")
    st.exception(e)
    st.stop()

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

# Ensure numeric price
df_el[PRICE_COL] = to_numeric_series(df_el[PRICE_COL])

# ---------------------------------------------------------------------
# Electricity stats / KPIs (full dataset)
# ---------------------------------------------------------------------
stats = compute_trailing_stats(df_el[PRICE_COL])

st.markdown("### ⚡ Electricity — Key statistics")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Mean (all)", fmt(stats["mean_all"], DECIMALS["price"], UNITS["price"]))
c3.metric("Std dev (all)", fmt(stats["std_all"], DECIMALS["price"], UNITS["price"]))

roll_df = pd.DataFrame(
    {
        "Window": ["Last 1 month", "Last 6 months", "Last 12 months"],
        f"Mean ({UNITS['price']})": [stats["1m_mean"], stats["6m_mean"], stats["12m_mean"]],
        f"Std dev ({UNITS['price']})": [stats["1m_std"], stats["6m_std"], stats["12m_std"]],
    }
)

roll_df[f"Mean ({UNITS['price']})"] = roll_df[f"Mean ({UNITS['price']})"].map(
    lambda v: fmt(v, DECIMALS["price"], UNITS["price"])
)
roll_df[f"Std dev ({UNITS['price']})"] = roll_df[f"Std dev ({UNITS['price']})"].map(
    lambda v: fmt(v, DECIMALS["price"], UNITS["price"])
)

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

# NEW: Weather city selector in sidebar (matches the new schema)
st.sidebar.markdown("---")
st.sidebar.subheader("Weather controls")
selected_city_label = st.sidebar.selectbox(
    "City",
    list(CITY_OPTIONS.keys()),
    index=list(CITY_OPTIONS.keys()).index(DEFAULT_CITY_LABEL),
)
selected_city = CITY_OPTIONS[selected_city_label]  # e.g. "stockholm"

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

# ---------------------------------------------------------------------
# IMPORTANT FIX:
# Compute moving averages on the FULL dataset FIRST (so they exist immediately
# when you zoom into a timeframe), using min_periods=1.
# Then filter for plotting.
# ---------------------------------------------------------------------
df_el_full = df_el.copy()

if show_ma_30:
    df_el_full["ma_30d"] = df_el_full[PRICE_COL].rolling(window=30, min_periods=1).mean()
if show_ma_182:
    df_el_full["ma_182d"] = df_el_full[PRICE_COL].rolling(window=182, min_periods=1).mean()
if show_ma_365:
    df_el_full["ma_365d"] = df_el_full[PRICE_COL].rolling(window=365, min_periods=1).mean()

df_el_plot = df_el_full[(df_el_full[TS_COL] >= start_ts) & (df_el_full[TS_COL] <= end_ts)].copy()
if df_el_plot.empty:
    st.warning("No electricity data in the selected timeframe.")
    st.stop()

# ---------------------------------------------------------------------
# Electricity plot
# ---------------------------------------------------------------------
st.subheader("📈 Historical electricity price")
st.caption(f"Unit: {UNITS['price']} • Timestamps shown in UTC")

fig_el = go.Figure()
fig_el.add_trace(
    go.Scatter(
        x=df_el_plot[TS_COL],
        y=df_el_plot[PRICE_COL],
        mode="lines",
        name=f"Price ({UNITS['price']})",
        line=dict(width=LINE_WIDTH, dash="solid"),
    )
)

if show_ma_30 and "ma_30d" in df_el_plot.columns:
    fig_el.add_trace(
        go.Scatter(
            x=df_el_plot[TS_COL],
            y=df_el_plot["ma_30d"],
            mode="lines",
            name=f"30d avg ({UNITS['price']})",
            line=dict(width=LINE_WIDTH, dash="solid"),
        )
    )

if show_ma_182 and "ma_182d" in df_el_plot.columns:
    fig_el.add_trace(
        go.Scatter(
            x=df_el_plot[TS_COL],
            y=df_el_plot["ma_182d"],
            mode="lines",
            name=f"6m avg ({UNITS['price']})",
            line=dict(width=LINE_WIDTH, dash="solid"),
        )
    )

if show_ma_365 and "ma_365d" in df_el_plot.columns:
    fig_el.add_trace(
        go.Scatter(
            x=df_el_plot[TS_COL],
            y=df_el_plot["ma_365d"],
            mode="lines",
            name=f"12m avg ({UNITS['price']})",
            line=dict(width=LINE_WIDTH, dash="solid"),
        )
    )

fig_el.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title=axis_title("Price", UNITS["price"]),
)

st.plotly_chart(fig_el, use_container_width=True)

st.subheader("📋 Latest electricity rows (filtered timeframe)")
display_el = df_el_plot.tail(50).copy()
display_el = display_el.rename(
    columns={
        TS_COL: "Date (UTC)",
        PRICE_COL: f"Price ({UNITS['price']})",
    }
)
st.dataframe(display_el, use_container_width=True)

# =====================================================================
# Weather section (UPDATED)
# =====================================================================
st.markdown("---")
st.header(f"🌦 Weather — {selected_city_label}")
st.caption(f"Weather feature group values used by the model ({selected_city_label}). All values are displayed with units.")

try:
    df_w = load_weather()
except Exception as e:
    st.error("Failed to load the weather feature group from Hopsworks.")
    st.exception(e)
    st.stop()

# Resolve city-specific column names
temp_col = TEMP_TPL.format(city=selected_city)
wind_col = WIND_TPL.format(city=selected_city)
precip_col = PRECIP_TPL.format(city=selected_city)
cloud_col = CLOUD_TPL.format(city=selected_city)

# Validate weather schema (timestamp + selected city columns)
missing_w = []
for col in [WEATHER_TS_COL, temp_col, wind_col, precip_col, cloud_col]:
    if col not in df_w.columns:
        missing_w.append(f"Missing weather column `{col}`")
if missing_w:
    st.error(
        "Weather schema mismatch. Update WEATHER_TS_COL or verify city-specific column names:\n- "
        + "\n- ".join(missing_w)
    )
    with st.expander("Developer: available weather columns"):
        st.write(list(df_w.columns))
    st.stop()

df_w = df_w.copy()
df_w[WEATHER_TS_COL] = pd.to_datetime(df_w[WEATHER_TS_COL], utc=True, errors="coerce")
df_w = df_w.dropna(subset=[WEATHER_TS_COL]).sort_values(WEATHER_TS_COL)

# Align to electricity timeframe selection
df_w_plot = df_w[(df_w[WEATHER_TS_COL] >= start_ts) & (df_w[WEATHER_TS_COL] <= end_ts)].copy()
if df_w_plot.empty:
    st.warning(f"No weather data found for {selected_city_label} in the selected timeframe.")
    st.stop()

# Ensure numeric columns
df_w_plot[temp_col] = to_numeric_series(df_w_plot[temp_col])
df_w_plot[wind_col] = to_numeric_series(df_w_plot[wind_col])
df_w_plot[precip_col] = to_numeric_series(df_w_plot[precip_col])
df_w_plot[cloud_col] = to_numeric_series(df_w_plot[cloud_col])

# Cloud normalization (0–1 -> 0–100)
cloud_raw = df_w_plot[cloud_col]
if is_prob_0_1(cloud_raw):
    cloud = cloud_raw * 100.0
else:
    cloud = cloud_raw

precip = df_w_plot[precip_col]

# Weather summary KPIs
st.markdown("### 🌦 Weather — Key statistics")

wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("Rows", f"{len(df_w_plot):,}")
wc2.metric("Precip mean", fmt(precip.mean(), DECIMALS["precip"], UNITS["precip"]) if precip.notna().any() else "—")
wc3.metric("Precip std", fmt(precip.std(), DECIMALS["precip"], UNITS["precip"]) if precip.notna().sum() > 1 else "—")
wc4.metric("Cloud mean", fmt(cloud.mean(), DECIMALS["cloud"], UNITS["cloud"]) if cloud.notna().any() else "—")

if cloud.notna().any():
    st.caption(
        f"Cloud cover range: {fmt(cloud.min(), DECIMALS['cloud'], UNITS['cloud'])} – "
        f"{fmt(cloud.max(), DECIMALS['cloud'], UNITS['cloud'])}"
    )
else:
    st.caption("Cloud cover not available.")

st.caption("Note: precipitation is shown in mm per time step (depends on how the feature group was created).")

# Temperature plot
st.subheader(f"🌡 Temperature (2m) — {selected_city_label}")
st.caption(f"Unit: {UNITS['temp']} • Timestamps shown in UTC")

fig_temp = go.Figure()
fig_temp.add_trace(
    go.Scatter(
        x=df_w_plot[WEATHER_TS_COL],
        y=df_w_plot[temp_col],
        mode="lines",
        name=f"Temperature (2m) ({UNITS['temp']})",
        line=dict(width=LINE_WIDTH, dash="solid"),
    )
)
fig_temp.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title=axis_title("Temperature (2m)", UNITS["temp"]),
)
st.plotly_chart(fig_temp, use_container_width=True)

# Wind plot
st.subheader(f"💨 Wind speed (10m) — {selected_city_label}")
st.caption(f"Unit: {UNITS['wind']} • Timestamps shown in UTC")

fig_wind = go.Figure()
fig_wind.add_trace(
    go.Scatter(
        x=df_w_plot[WEATHER_TS_COL],
        y=df_w_plot[wind_col],
        mode="lines",
        name=f"Wind speed (10m) ({UNITS['wind']})",
        line=dict(width=LINE_WIDTH, dash="solid"),
    )
)
fig_wind.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Time (UTC)",
    yaxis_title=axis_title("Wind speed (10m)", UNITS["wind"]),
)
st.plotly_chart(fig_wind, use_container_width=True)

st.subheader(f"📋 Latest weather rows ({selected_city_label}, filtered timeframe)")

# Build a clean display table with units in column headers
display_w = df_w_plot[[WEATHER_TS_COL, temp_col, wind_col, precip_col, cloud_col]].tail(50).copy()
display_w["cloud_cover_display"] = cloud.loc[display_w.index]

display_w = display_w.rename(
    columns={
        WEATHER_TS_COL: "Date (UTC)",
        temp_col: f"Temperature (2m) ({UNITS['temp']})",
        wind_col: f"Wind speed (10m) ({UNITS['wind']})",
        precip_col: f"Precipitation ({UNITS['precip']})",
        "cloud_cover_display": f"Cloud cover ({UNITS['cloud']})",
    }
)

# Drop the raw cloud col if you prefer (since cloud_cover_display is normalized)
# display_w = display_w.drop(columns=[cloud_col], errors="ignore")

st.dataframe(display_w, use_container_width=True)
