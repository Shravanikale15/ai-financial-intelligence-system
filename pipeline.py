import joblib
import pandas as pd
import numpy as np
# ----------------------------
# Load Models Once
# ----------------------------

health_model = joblib.load("saved_models/health_score_model.pkl")
health_features = joblib.load("saved_models/health_score_features.pkl")

stress_model = joblib.load("saved_models/stress_model.pkl")
stress_scaler = joblib.load("saved_models/stress_scaler.pkl")
stress_features = joblib.load("saved_models/stress_features.pkl")

personality_model = joblib.load("saved_models/personality_model.pkl")
personality_scaler = joblib.load("saved_models/personality_scaler.pkl")
personality_features = joblib.load("saved_models/personality_features.pkl")

def calculate_risk_score(health_score, stress_level, personality):

    # Convert stress to numeric
    stress_map = {
        "Low": 0,
        "Medium": 50,
        "High": 100
    }

    stress_numeric = stress_map.get(stress_level, 50)

    # Personality aggressiveness score
    personality_map = {
        "Saver": 30,
        "Stable": 50,
        "Risk-Oriented": 80,
        "Impulsive": 70
    }

    personality_score = personality_map.get(personality, 50)

    # Final risk score formula
    risk_score = (
        0.5 * health_score +
        0.3 * personality_score -
        0.2 * stress_numeric
    )

    return round(risk_score, 2)
def suggest_allocation(risk_score):

    if risk_score < 40:
        return {
            "Equity": "20%",
            "Debt": "60%",
            "Gold": "20%"
        }

    elif risk_score < 70:
        return {
            "Equity": "40%",
            "Debt": "40%",
            "Gold": "20%"
        }

    else:
        return {
            "Equity": "70%",
            "Debt": "20%",
            "Gold": "10%"
        }

def evaluate_goal_feasibility(goal_amount, projection):

    cumulative_savings = 0
    months_needed = 0

    for value in projection:
        cumulative_savings += value
        months_needed += 1
        
        if cumulative_savings >= goal_amount:
            return {
                "achievable": True,
                "months_needed": months_needed,
                "shortfall": 0
            }

    shortfall = goal_amount - cumulative_savings

    return {
        "achievable": False,
        "months_needed": None,
        "shortfall": round(float(shortfall), 2)
    }
def generate_projection(current_surplus, months=6):
    
    projections = []
    rolling_value = current_surplus
    
    for i in range(months):
        # Slight stabilization factor
        rolling_value = rolling_value * 0.98  # 2% conservative adjustment
        projections.append(round(float(rolling_value), 2))
    
    return projections

def run_full_financial_analysis(user_input: dict):

    user_df = pd.DataFrame([user_input])

    # --------------------------
    # Health Prediction
    # --------------------------
    health_df = user_df[health_features]
    health_score = health_model.predict(health_df)[0]
    user_df["health_score"] = health_score

    # --------------------------
    # Stress Prediction
    # --------------------------
    stress_df = user_df[stress_features]
    stress_scaled = stress_scaler.transform(stress_df)
    stress_level = stress_model.predict(stress_scaled)[0]

    # --------------------------
    # Personality Prediction
    # --------------------------
    personality_df = user_df[personality_features]
    personality_scaled = personality_scaler.transform(personality_df)
    cluster = personality_model.predict(personality_scaled)[0]

    cluster_map = {
        0: "Saver",
        1: "Impulsive",
        2: "Stable",
        3: "Risk-Oriented"
    }

    personality = cluster_map.get(cluster, "Unknown")

    # --------------------------
    # Savings Projection
    # --------------------------
    current_surplus = user_input.get("monthly_surplus", 0)
    projection = generate_projection(current_surplus)


    goal_amount = user_input.get("goal_amount")

    goal_result = None

    if goal_amount:
        goal_result = evaluate_goal_feasibility(goal_amount, projection)
    
    risk_score = calculate_risk_score(health_score, stress_level, personality)
    allocation = suggest_allocation(risk_score)

    # --------------------------
    # Return Combined Result
    # --------------------------
    return {
    "health_score": round(float(health_score), 2),
    "stress_level": stress_level,
    "personality": personality,
    "6_month_projection": projection,
    "goal_analysis": goal_result,
    "risk_score": risk_score,
    "investment_allocation": allocation
}