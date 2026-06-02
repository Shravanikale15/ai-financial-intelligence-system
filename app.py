from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pipeline import run_full_financial_analysis
import database

app = Flask(__name__)
CORS(app)

# Initialize database
database.init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        result = run_full_financial_analysis(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required"}), 400
        
    res = database.register_user(email, password)
    if res["success"]:
        return jsonify({"success": True, "message": "User registered successfully"})
    else:
        return jsonify({"success": False, "error": res["error"]}), 400


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required"}), 400
        
    user = database.authenticate_user(email, password)
    if user:
        state = database.get_user_state(user["id"])
        return jsonify({
            "success": True,
            "user": user,
            "state": state
        })
    else:
        return jsonify({"success": False, "error": "Invalid email or password"}), 401


@app.route("/sync", methods=["POST"])
def sync():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    state = data.get("state")
    
    if not user_id or state is None:
        return jsonify({"success": False, "error": "user_id and state are required"}), 400
        
    success = database.save_user_state(user_id, state)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to sync state"}), 500


if __name__ == "__main__":
    app.run(debug=True)