import joblib
import pandas as pd

print("Loading models...")


# -----------------------------
# Load Health Model
# -----------------------------
health_model = joblib.load("saved_models/health_score_model.pkl")
health_features = joblib.load("saved_models/health_score_features.pkl")

# -----------------------------
# Load Stress Model
# -----------------------------
stress_model = joblib.load("saved_models/stress_model.pkl")
stress_scaler = joblib.load("saved_models/stress_scaler.pkl")
stress_features = joblib.load("saved_models/stress_features.pkl")

print("Models loaded successfully")


# =============================
# Example User Input
# =============================
user_input = {
    "savings_rate": 0.35,
    "expense_ratio": 0.55,
    "income_variance": 15000,
    "expense_volatility": 3000,
    "monthly_surplus": 20000,
    "expense_spike": 500
}

# Convert to DataFrame
user_df = pd.DataFrame([user_input])


# =============================
# STEP 1: Predict Health Score
# =============================
health_df = user_df[health_features]

health_score = health_model.predict(health_df)[0]

print("\nPredicted Health Score:", round(health_score, 2))


# Add health score for stress model
user_df["health_score"] = health_score


# =============================
# STEP 2: Predict Stress Level
# =============================
stress_df = user_df[stress_features]

# Scale before prediction
stress_scaled = stress_scaler.transform(stress_df)

stress_level = stress_model.predict(stress_scaled)[0]

print("Predicted Stress Level:", stress_level)