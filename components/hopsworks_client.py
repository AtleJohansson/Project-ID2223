import os
import streamlit as st
import hopsworks

@st.cache_resource
def get_feature_store():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key or not project_name:
        raise ValueError("Missing HOPSWORKS_API_KEY or HOPSWORKS_PROJECT")

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        host=host,
    )
    return project.get_feature_store()
