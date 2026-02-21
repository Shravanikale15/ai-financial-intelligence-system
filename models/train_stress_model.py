import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv("data/processed/master_dataset.csv")

print("Dataset loaded successfully")

# -------------------------------------
# STEP 1: Load Health Score Model
# -------------------------------------
health_model = joblib.load("saved_models/health_score_model.pkl")
health_features = joblib.load("saved_models/health_score_features.pkl")

df["health_score"] = health_model.predict(df[health_features])

print("Health scores generated")

# -------------------------------------
# STEP 2: Create Stress Labels
# -------------------------------------

# Normalize required columns
df["surplus_norm"] = df["monthly_surplus"] / df["monthly_surplus"].abs().max()
df["spike_norm"] = df["expense_spike"] / df["expense_spike"].abs().max()
df["volatility_norm"] = df["expense_volatility"] / df["expense_volatility"].max()

# Create stress score
df["stress_score"] = (
    0.5 * (1 - df["health_score"]/100) +
    0.3 * (1 - df["surplus_norm"]) +
    0.1 * df["spike_norm"] +
    0.1 * df["volatility_norm"]
)

# Create stress_score same as before
df["stress_score"] = (
    0.5 * (1 - df["health_score"]/100) +
    0.3 * (1 - df["surplus_norm"]) +
    0.1 * df["spike_norm"] +
    0.1 * df["volatility_norm"]
)

# Use percentiles to create balanced classes
low_thresh = df["stress_score"].quantile(0.33)
high_thresh = df["stress_score"].quantile(0.66)

def assign_stress(score):
    if score <= low_thresh:
        return "Low"
    elif score <= high_thresh:
        return "Medium"
    else:
        return "High"

df["stress_level"] = df["stress_score"].apply(assign_stress)

print("Stress labels created")

# -------------------------------------
# STEP 3: Prepare Features
# -------------------------------------

features = [
    "health_score",
    "monthly_surplus",
    "expense_spike",
    "expense_volatility"
]

X = df[features]
y = df["stress_level"]

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------
# STEP 4: Train Logistic Regression
# -------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Stress model trained")

# -------------------------------------
# STEP 5: Evaluate
# -------------------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------------
# STEP 6: Save Model + Scaler
# -------------------------------------

joblib.dump(model, "saved_models/stress_model.pkl")
joblib.dump(scaler, "saved_models/stress_scaler.pkl")
joblib.dump(features, "saved_models/stress_features.pkl")

print("Stress model saved successfully")