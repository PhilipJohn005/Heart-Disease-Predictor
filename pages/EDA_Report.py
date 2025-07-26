import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="EDA Report", layout="wide")
st.title("📊 Heart Disease EDA Report")

# Load and profile data
data = pd.read_csv("./Data/heart_disease_uci.csv")
data = data[data['num'].isin([0, 1])]
data['num'] = data['num'].astype(int)

# Create and show report
with st.spinner("Generating profile report..."):
    profile = ProfileReport(data, title="Heart Disease EDA", minimal=True)
    st_profile_report(profile)
