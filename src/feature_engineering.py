import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_transactions.csv")

print("Loaded cleaned dataset")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Extract year and month
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Separate income and expense
df['income'] = df['amount'].where(df['transaction_type'] == 'income', 0)
df['expense'] = df['amount'].where(df['transaction_type'] == 'expense', 0)

# Monthly aggregation
monthly = df.groupby(['year', 'month', 'user_id']).agg({
    'income': 'sum',
    'expense': 'sum'
}).reset_index()

# Create savings
monthly['savings'] = monthly['income'] - monthly['expense']

# Create ratios
monthly['savings_rate'] = monthly['savings'] / monthly['income']
monthly['expense_ratio'] = monthly['expense'] / monthly['income']

# Replace infinite and NaN
monthly.replace([float('inf'), -float('inf')], 0, inplace=True)
monthly.fillna(0, inplace=True)

# Save file
monthly.to_csv("data/processed/monthly_financial_features.csv", index=False)

print("Feature engineering completed and saved")
