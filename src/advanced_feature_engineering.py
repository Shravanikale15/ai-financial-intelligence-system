import pandas as pd

# Load monthly features
df = pd.read_csv("data/processed/monthly_financial_features.csv")

# Sort values
df = df.sort_values(by=["user_id", "year", "month"])

# Calculate income variance
df["income_variance"] = df.groupby("user_id")["income"].transform("std")

# Calculate expense volatility
df["expense_volatility"] = df.groupby("user_id")["expense"].transform("std")

# Monthly surplus
df["monthly_surplus"] = df["savings"]

# Expense spike (current expense vs rolling avg)
df["expense_rolling_mean"] = df.groupby("user_id")["expense"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df["expense_spike"] = df["expense"] - df["expense_rolling_mean"]

# Replace NaN
df.fillna(0, inplace=True)

# Save file
df.to_csv("data/processed/final_engineered_features.csv", index=False)

print("Advanced feature engineering completed")
