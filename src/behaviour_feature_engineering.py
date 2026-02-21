import pandas as pd

# Load original cleaned transactions
df = pd.read_csv("data/processed/cleaned_transactions.csv")

expense_df = df[df["transaction_type"] == "expense"].copy()

# Fix negative values
expense_df["amount"] = expense_df["amount"].abs()

# Total expense per user
total_expense = expense_df.groupby("user_id")["amount"].sum()

# Food expense
food = expense_df[expense_df["category"] == "food"].groupby("user_id")["amount"].sum()

# Entertainment expense
entertainment = expense_df[expense_df["category"] == "entertainment"].groupby("user_id")["amount"].sum()

# Other discretionary
others = expense_df[expense_df["category"].isin(["others"])].groupby("user_id")["amount"].sum()

# Create dataframe
behaviour = pd.DataFrame({
    "total_expense": total_expense,
    "food_expense": food,
    "entertainment_expense": entertainment,
    "discretionary_expense": others
}).fillna(0)

# Calculate ratios
behaviour["food_ratio"] = behaviour["food_expense"] / behaviour["total_expense"]
behaviour["entertainment_ratio"] = behaviour["entertainment_expense"] / behaviour["total_expense"]
behaviour["discretionary_ratio"] = behaviour["discretionary_expense"] / behaviour["total_expense"]

behaviour.fillna(0, inplace=True)

# Save
behaviour.to_csv("data/processed/behaviour_features.csv")

print("Behaviour features created")
