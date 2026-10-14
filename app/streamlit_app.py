"""
Simple Streamlit dashboard for the churn model.
I wanted something that a non-technical manager could actually use.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append("src")

from data_preprocessing import load_raw_data, create_features

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

st.title("Telecom Customer Churn Analysis")
st.markdown("پیش‌بینی احتمال ریزش مشتری + تحلیل کلی")

# ---------- load data once ----------
@st.cache_data
def get_data():
    raw = load_raw_data("../data/raw/telco_customer_churn.csv")
    df = create_features(raw)
    return df

df = get_data()

# ---------- sidebar filters ----------
st.sidebar.header("Filters / فیلترها")
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique() if "Contract" in df.columns else [],
    default=None
)

# ---------- KPI cards ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{len(df):,}")
with col2:
    churn_rate = df["Churn"].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
with col3:
    avg_tenure = df["tenure"].mean()
    st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
with col4:
    avg_monthly = df["MonthlyCharges"].mean()
    st.metric("Avg Monthly Charge", f"${avg_monthly:.1f}")

st.markdown("---")

# ---------- charts ----------
left, right = st.columns(2)

with left:
    st.subheader("Churn by Contract Type")
    # simple value counts
    if "Contract" in df.columns:
        contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(contract_churn)

with right:
    st.subheader("Churn by Tenure Group")
    if "tenure_group" in df.columns:
        tenure_churn = df.groupby("tenure_group")["Churn"].mean()
        st.bar_chart(tenure_churn)

st.markdown("---")
st.subheader("Quick Prediction (what-if)")

# a few input widgets so someone can play with a single customer
c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
with c2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with c3:
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

st.info("این فقط یک نمونه تعاملی ساده‌ست. برای پیش‌بینی واقعی از مدل ذخیره‌شده استفاده کنید.")

st.markdown("---")
st.caption("Built for portfolio purposes – data from IBM Telco Customer Churn dataset")
