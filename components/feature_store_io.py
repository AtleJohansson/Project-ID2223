import pandas as pd
import streamlit as st
from components.hopsworks_client import get_feature_store

@st.cache_data(ttl=300)
def read_feature_group(name: str, version: int) -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(name=name, version=version)
    return fg.read()
