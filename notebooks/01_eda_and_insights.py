"""
Exploratory Data Analysis – Telecom Customer Churn
I went through the data step by step and noted the things that looked important for the business.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../src")

from data_preprocessing import load_raw_data, create_features

# style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

raw = load_raw_data("../data/raw/telco_customer_churn.csv")
df = create_features(raw)

print("=" * 50)
print("BASIC OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nChurn rate: {df['Churn'].mean()*100:.1f}%")
print("\nMissing values left:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n" + "=" * 50)
print("KEY BUSINESS OBSERVATIONS")
print("=" * 50)

# 1. Contract type is the strongest signal I found
print("\n1. Churn by Contract:")
print(df.groupby("Contract")["Churn"].agg(["mean", "count"]).round(3))

# 2. Tenure
print("\n2. Churn by Tenure Group:")
print(df.groupby("tenure_group")["Churn"].mean().round(3))

# 3. Internet service
print("\n3. Churn by Internet Service:")
print(df.groupby("InternetService")["Churn"].mean().round(3))

# 4. Payment method
print("\n4. Churn by Payment Method:")
print(df.groupby("PaymentMethod")["Churn"].mean().round(3))

# 5. Number of services
print("\n5. Churn by number of additional services:")
print(df.groupby("num_services")["Churn"].mean().round(3))

print("\n" + "=" * 50)
print("QUICK CORRELATIONS (numeric)")
print("=" * 50)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly", "Churn"]
print(df[num_cols].corr()["Churn"].sort_values())

# Save a couple of simple plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df.groupby("Contract")["Churn"].mean().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Churn Rate by Contract Type")
axes[0].set_ylabel("Churn Rate")
axes[0].tick_params(axis="x", rotation=15)

df.groupby("tenure_group")["Churn"].mean().plot(kind="bar", ax=axes[1], color="darkorange")
axes[1].set_title("Churn Rate by Tenure Group")
axes[1].set_ylabel("Churn Rate")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("../reports/figures/churn_by_contract_tenure.png", dpi=120, bbox_inches="tight")
print("\nPlot saved to reports/figures/churn_by_contract_tenure.png")
plt.close()

print("\nEDA finished.")
