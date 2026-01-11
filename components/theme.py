import streamlit as st

DARK_ONLY_CSS = """
<style>
:root{
  --bg: #0b1220;
  --text: #e5e7eb;
  --muted: #a1a1aa;
  --card: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.10);
  --link: #60a5fa;
}

.stApp { background: var(--bg); color: var(--text); }
a { color: var(--link) !important; }
hr { border-color: var(--border); }

.block-container { padding-top: 2rem; }

.section-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 18px;
  margin: 10px 0 18px 0;
}
.section-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 6px; }
.section-muted { color: var(--muted); font-size: 0.95rem; }

.kpi {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
}

/* Sidebar polish */
section[data-testid="stSidebar"] > div {
  background: #0b1220;
}
</style>
"""

def apply_dark_only() -> None:
    st.markdown(DARK_ONLY_CSS, unsafe_allow_html=True)

def plotly_template_dark() -> str:
    return "plotly_dark"
