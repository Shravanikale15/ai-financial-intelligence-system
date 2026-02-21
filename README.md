# AI Financial Intelligence System

An end-to-end Machine Learning powered financial analytics system that evaluates financial health, stress level, behavioral personality, future projections, goal feasibility, and investment allocation.

---

##  Problem Statement

Individuals lack structured insights into their financial stability and future sustainability.  
This system builds an AI-driven intelligence layer to analyze financial patterns and generate actionable insights.

---

## Features Implemented

### 1]Financial Health Score (Regression)
- Model: Random Forest Regressor
- Output: Score (0–100)
- Evaluates overall financial stability

### 2️] Financial Stress Index (Classification)
- Model: Logistic Regression
- Output: Low / Medium / High
- Detects financial pressure levels

### 3️] Spending Personality (Clustering)
- Model: KMeans
- Output: Saver / Stable / Impulsive / Risk-Oriented
- Behavioral segmentation

### 4️] 6-Month Savings Projection
- Method: Rolling Projection
- Predicts future monthly surplus

### 5️] Goal Feasibility Tracker
- Calculates months required to reach financial goal
- Determines shortfall if any

### 6️] Investment Allocation Engine
- Computes risk score
- Suggests Equity / Debt / Gold allocation

---

## 🏗 System Architecture

User Input → Feature Engineering → ML Models → Intelligence Layer → Dashboard Output

---

## 🛠 Tech Stack

- Python
- Scikit-Learn
- Pandas
- Flask
- Bootstrap
- Chart.js

---
## 📦 How to Run

```bash
pip install -r requirements.txt
python app.py


📊 Sample API Request
{
  "savings_rate": 0.35,
  "expense_ratio": 0.55,
  "income_variance": 15000,
  "expense_volatility": 3000,
  "monthly_surplus": 20000,
  "expense_spike": 500,
  "food_ratio": 0.2,
  "entertainment_ratio": 0.1,
  "discretionary_ratio": 0.15,
  "goal_amount": 100000
}
📈 Sample Output
{
  "health_score": 94.9,
  "stress_level": "Low",
  "personality": "Saver",
  "6_month_projection": [...],
  "goal_analysis": {...},
  "investment_allocation": {...}
}
🔮 Future Improvements

Replace rule-based allocation with ML classifier

Deploy on cloud

Add authentication system

Add real-time financial data integration