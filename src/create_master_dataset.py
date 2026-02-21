import pandas as pd

# Load monthly engineered features
monthly = pd.read_csv("data/processed/final_engineered_features.csv")

# Load behaviour features
behaviour = pd.read_csv("data/processed/behaviour_features.csv")

# Merge on user_id
master = monthly.merge(behaviour, on="user_id", how="left")

# Fill missing values
master.fillna(0, inplace=True)

# Save master dataset
master.to_csv("data/processed/master_dataset.csv", index=False)

print("MASTER DATASET CREATED")
print(master.head())
