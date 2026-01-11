from dataclasses import dataclass
from typing import Callable, Optional
import streamlit as st
import pandas as pd

@dataclass
class Section:
    title: str
    body_md: str = ""
    caption: str = ""
    chart_fn: Optional[Callable[[], None]] = None   # chart_fn writes chart(s)
    table_df: Optional[pd.DataFrame] = None         # optional table
    table_caption: str = ""

def render_section(s: Section) -> None:
    st.markdown(
        f"""
        <div class="section-card">
          <div class="section-title">{s.title}</div>
          <div class="section-muted">{s.caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if s.body_md:
        st.markdown(s.body_md)

    if s.chart_fn is not None:
        with st.container():
            s.chart_fn()

    if s.table_df is not None:
        if s.table_caption:
            st.caption(s.table_caption)
        st.dataframe(s.table_df, use_container_width=True)

    st.markdown("---")

def render_blog(sections: list[Section]) -> None:
    for s in sections:
        render_section(s)
