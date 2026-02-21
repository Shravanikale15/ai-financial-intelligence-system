import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv("data/processed/master_dataset.csv")
df["monthly_surplus"] = df["monthly_surplus"].clip(-50000, 100000)

# Sort properly
df = df.sort_values(["user_id", "year", "month"])

# Create next month target
df["next_month_savings"] = df.groupby("user_id")["monthly_surplus"].shift(-1)

# Drop last month rows (no target)
df = df.dropna(subset=["next_month_savings"])

features = [
    "savings_rate",
    "expense_ratio",
    "expense_volatility",
    "monthly_surplus"
]

X = df[features]
y = df["next_month_savings"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)

# Save
joblib.dump(model, "saved_models/projection_model.pkl")
joblib.dump(scaler, "saved_models/projection_scaler.pkl")
joblib.dump(features, "saved_models/projection_features.pkl")

print("Projection model saved successfully")