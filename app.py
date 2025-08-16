# app.py
# A minimal Flask API for Boston Housing price prediction using a joblib-saved model.

from pathlib import Path
import joblib                     # for loading the trained scikit-learn model
import numpy as np               # for shaping numeric input
from flask import Flask, request, jsonify  # web framework

app = Flask(__name__)

# ---------- Load model (once, at startup) ----------
# Resolve the model path relative to this file so it works when run from anywhere.
MODEL_PATH = Path(__file__).resolve().parent / "boston_linreg_model.pkl"

# Load the pre-trained estimator; this must match the version used at training time.
# If this line raises, check that the .pkl exists and that sklearn/joblib versions are compatible.
model = joblib.load(MODEL_PATH)

# ---------- Feature schema ----------
# IMPORTANT: This must match the exact training column order used to fit the model.
FEATURE_ORDER = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

def to_feature_row(payload: dict) -> np.ndarray:
    """
    Convert a dict of features into a 2D numpy array in the required order.
    - payload: dictionary under the "data" key from the request JSON
    Returns: shape (1, n_features) float array ready for model.predict(...)
    """
    # Ensure every required key is present; raise a helpful error if one is missing.
    missing = [k for k in FEATURE_ORDER if k not in payload]
    if missing:
        raise KeyError(f"Missing feature(s): {', '.join(missing)}")
    # Build row in deterministic order and cast to float; reshape to (1, -1)
    row = [payload[k] for k in FEATURE_ORDER]
    return np.asarray(row, dtype=float).reshape(1, -1)

# ---------- Routes ----------
@app.get("/")
def health():
    """Simple health/docs endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Boston Housing Prediction API",
        "predict_endpoint": "/predict_api",
        "expected_json_schema": {
            "data": {k: "float|int" for k in FEATURE_ORDER}
        }
    }), 200

@app.post("/predict_api")
def predict_api():
    """
    Accepts JSON like:
    {
      "data": {
        "CRIM": 0.1, "ZN": 0, "INDUS": 8.0, "CHAS": 0, "NOX": 0.5,
        "RM": 6.0, "AGE": 65, "DIS": 4.0, "RAD": 4, "TAX": 300,
        "PTRATIO": 18.0, "B": 390.0, "LSTAT": 12.0
      }
    }
    Returns: {"prediction": <float>}  # predicted MEDV in $1000s
    """
    try:
        body = request.get_json(silent=True) or {}
        features = body.get("data")
        if not isinstance(features, dict):
            return jsonify({"error": "JSON must include key 'data' with a feature dict."}), 400

        X = to_feature_row(features)                # validate and vectorize
        y_pred = model.predict(X)                   # scikit-learn predict
        return jsonify({"prediction": float(y_pred[0])}), 200

    except KeyError as e:
        # A required feature was missing
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        # Bad types/shape conversions
        return jsonify({"error": f"Invalid values: {e}"}), 400
    except Exception as e:
        # Catch-all for unexpected errors
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# ---------- Entry point ----------
if __name__ == "__main__":
    # debug=True enables auto-reload + helpful tracebacks in development
    app.run(debug=True)  # default: http://127.0.0.1:5000

    