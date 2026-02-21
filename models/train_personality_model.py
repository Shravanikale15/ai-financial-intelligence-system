import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv("data/processed/master_dataset.csv")

print("Dataset loaded successfully")

# -----------------------------
# STEP 1: Select Features
# -----------------------------
features = [
    "savings_rate",
    "food_ratio",
    "entertainment_ratio",
    "discretionary_ratio",
    "expense_volatility"
]

X = df[features].dropna()

# -----------------------------
# STEP 2: Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data scaled")

# -----------------------------
# STEP 3: Train KMeans
# -----------------------------
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

print("KMeans trained")

# -----------------------------
# STEP 4: Save Model
# -----------------------------
joblib.dump(kmeans, "saved_models/personality_model.pkl")
joblib.dump(scaler, "saved_models/personality_scaler.pkl")
joblib.dump(features, "saved_models/personality_features.pkl")

print("Personality model saved successfully")