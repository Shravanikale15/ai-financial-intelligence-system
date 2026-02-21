from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pipeline import run_full_financial_analysis

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        result = run_full_financial_analysis(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)