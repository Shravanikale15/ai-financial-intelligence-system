import joblib
import pandas as pd

# Load model and features
model = joblib.load("saved_models/health_score_model.pkl")
features = joblib.load("saved_models/health_score_features.pkl")

# Create input as DataFrame
new_user = pd.DataFrame([{
    "savings_rate": 0.35,
    "expense_ratio": 0.55,
    "income_variance": 15000,
    "expense_volatility": 3000,
    "monthly_surplus": 20000,
    "expense_spike": 500
}])

# Ensure correct column order
new_user = new_user[features]

# Predict
health_score = model.predict(new_user)

print("Predicted Health Score:", round(health_score[0], 2))