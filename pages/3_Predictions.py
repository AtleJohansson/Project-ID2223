import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_app.components.io import read_parquet_s3
from streamlit_app.components.theme import theme_toggle_sidebar, apply_theme_css, plotly_template
from streamlit_app.components.sections import Section, render_blog

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

theme = theme_toggle_sidebar()
apply_theme_css(theme)

st.title("🔮 Latest Forecast")

bucket = os.environ.get("S3_BUCKET")
if not bucket:
    st.error("S3_BUCKET not set. Add it to your .env or Streamlit secrets.")
    st.stop()

prefix = os.environ.get("S3_PREFIX", "").strip("/")
pred_key = f"{prefix}/forecast_latest.parquet" if prefix else "forecast_latest.parquet"
history_key = f"{prefix}/history.parquet" if prefix else "history.parquet"

@st.cache_data(ttl=300)
def load_pred():
    return read_parquet_s3(bucket, pred_key)

@st.cache_data(ttl=300)
def load_hist_tail(n=2000):
    df = read_parquet_s3(bucket, history_key)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.tail(n)

pred = load_pred()
hist = load_hist_tail()

# Normalize schema
if "timestamp" not in pred.columns:
    st.error("Expected a `timestamp` column in forecast_latest.parquet.")
    st.stop()
pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
pred = pred.dropna(subset=["timestamp"]).sort_values("timestamp")

pred_col = "predicted_price"
if pred_col not in pred.columns:
    st.error("Expected `predicted_price` column in forecast_latest.parquet.")
    st.stop()

if "price" not in hist.columns:
    st.error("Expected `price` column in history.parquet.")
    st.stop()

def forecast_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["price"],
        mode="lines", name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=pred["timestamp"], y=pred[pred_col],
        mode="lines", name="Forecast"
    ))

    if "yhat_lower" in pred.columns and "yhat_upper" in pred.columns:
        fig.add_trace(go.Scatter(
            x=pred["timestamp"].tolist() + pred["timestamp"].tolist()[::-1],
            y=pred["yhat_upper"].tolist() + pred["yhat_lower"].tolist()[::-1],
            fill="toself",
            line=dict(width=0),
            name="Uncertainty",
            showlegend=True,
        ))

    fig.update_layout(
        template=plotly_template(theme),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

# “nice table” view
table = pred.copy()
# optional formatting
for c in table.columns:
    if c != "timestamp" and pd.api.types.is_numeric_dtype(table[c]):
        table[c] = table[c].round(3)

sections = [
    Section(
        title="Forecast",
        caption="Latest model prediction for the next horizon",
        body_md="""
This chart shows:
- the most recent historical prices (tail window)
- the forecast trajectory
- optional uncertainty bands if provided
""",
        chart_fn=forecast_chart,
    ),
    Section(
        title="Forecast table",
        caption="Easy to copy/paste for reporting or debugging",
        body_md="",
        table_df=table,
    ),
]

render_blog(sections)
