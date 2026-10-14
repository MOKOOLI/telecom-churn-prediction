"""
Data preprocessing utilities for the telecom churn project.
I wrote these helper functions while exploring the dataset so I can reuse them later.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_raw_data(path="data/raw/telco_customer_churn.csv"):
    """Load the original CSV and do the most basic cleaning."""
    df = pd.read_csv(path)
    
    # TotalCharges has some empty strings that should be treated as missing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop the few rows where TotalCharges is null (they have tenure = 0 anyway)
    df = df.dropna(subset=["TotalCharges"]).copy()
    
    # customerID is useless for modeling
    df = df.drop(columns=["customerID"])
    
    return df


def create_features(df):
    """
    Feature engineering based on what I saw during EDA.
    Some of these ideas came from looking at tenure and contract type together.
    """
    df = df.copy()
    
    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    
    # Tenure groups – I found that people in the first year churn a lot more
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"]
    )
    
    # Average monthly spend relative to total
    # (helps catch customers who suddenly increased usage)
    df["avg_monthly"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # Number of additional services
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # These columns have "No internet service" which we treat as 0
    for col in service_cols:
        df[col + "_flag"] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})
    
    df["num_services"] = df[[c + "_flag" for c in service_cols]].sum(axis=1)
    
    # Has both streaming services
    df["both_streaming"] = (
        (df["StreamingTV_flag"] == 1) & (df["StreamingMovies_flag"] == 1)
    ).astype(int)
    
    return df


def encode_and_split(df, test_size=0.2, random_state=42):
    """
    Encode categoricals and split into train/test.
    I keep the original categoricals for some plots later.
    """
    df = df.copy()
    
    # Columns that are still categorical
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Label encode everything for tree models (they handle it fine)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    # quick test
    raw = load_raw_data()
    print("Raw shape:", raw.shape)
    featured = create_features(raw)
    print("After features:", featured.shape)
    print(featured["Churn"].value_counts(normalize=True))
