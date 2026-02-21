import joblib
import pandas as pd

print("Loading personality model...")

model = joblib.load("saved_models/personality_model.pkl")
scaler = joblib.load("saved_models/personality_scaler.pkl")
features = joblib.load("saved_models/personality_features.pkl")

print("Model loaded")

# Example input
new_user = pd.DataFrame([{
    "savings_rate": 0.4,
    "food_ratio": 0.2,
    "entertainment_ratio": 0.1,
    "discretionary_ratio": 0.15,
    "expense_volatility": 3000
}])

X_scaled = scaler.transform(new_user[features])
cluster = model.predict(X_scaled)[0]

# Map cluster to personality
cluster_map = {
    0: "Saver",
    1: "Impulsive",
    2: "Stable",
    3: "Risk-Oriented"
}

personality = cluster_map.get(cluster, "Unknown")

print("Predicted Personality:", personality)