import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

print("Loading dataset...")

# Load master dataset
df = pd.read_csv("data/processed/master_dataset.csv")

print("Dataset loaded successfully")

# -----------------------------
# STEP 1: Create Target Variable
# -----------------------------
# We create health score from financial indicators
# Normalize variance columns first
df["income_variance_norm"] = df["income_variance"] / df["income_variance"].max()
df["expense_volatility_norm"] = df["expense_volatility"] / df["expense_volatility"].max()

# Create health score
df["health_score"] = (
    0.35 * df["savings_rate"] +
    0.25 * (1 - df["expense_ratio"]) +
    0.20 * (1 - df["expense_volatility_norm"]) +
    0.20 * (1 - df["income_variance_norm"])
) * 100

# Scale health score to 0–100
scaler = MinMaxScaler(feature_range=(0, 100))
df["health_score"] = scaler.fit_transform(df[["health_score"]])

print("Health score created")

# -----------------------------
# STEP 2: Select Features
# -----------------------------
features = [
    "savings_rate",
    "expense_ratio",
    "income_variance",
    "expense_volatility",
    "monthly_surplus",
    "expense_spike"
]

X = df[features]
y = df["health_score"]

# -----------------------------
# STEP 3: Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train-Test split completed")

# -----------------------------
# STEP 4: Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")

# -----------------------------
# STEP 5: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

# -----------------------------
# STEP 6: Save Model
# -----------------------------
joblib.dump(model, "saved_models/health_score_model.pkl")
joblib.dump(features, "saved_models/health_score_features.pkl")

print("Model saved successfully")
