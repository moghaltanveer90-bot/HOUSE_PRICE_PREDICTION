
"""
Project: End-to-End House Price Prediction
Author: Moghal Tanveer

Description:
This Flask application connects the trained XGBoost machine learning
model with a web interface and predicts house prices using
user-provided house details.
"""

import os
import joblib
import pandas as pd

from flask import Flask, render_template, request


# --------------------------------------------------
# 1. INITIALIZE FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "house_price_model.pkl"
)

# These are the exact feature columns used by main.py.
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "statezip",
    "country"
]

NUMERIC_FIELDS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated"
]

CATEGORICAL_FIELDS = [
    "city",
    "statezip",
    "country"
]


# --------------------------------------------------
# 2. LOAD TRAINED MACHINE LEARNING MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Please run main.py first."
    )

model = joblib.load(MODEL_PATH)

print("House Price Prediction Model Loaded Successfully!")


# --------------------------------------------------
# 3. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    """Display the house price prediction homepage."""

    return render_template("index.html")


# --------------------------------------------------
# 4. HOUSE PRICE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Collect house details, predict the price,
    and display the prediction on the homepage.
    """

    house_data = {}

    try:
        # Read and validate numerical inputs.
        for field in NUMERIC_FIELDS:
            value = request.form.get(field, "").strip()

            if not value:
                raise ValueError(f"Please enter {field}.")

            house_data[field] = float(value)

        # Read categorical inputs.
        for field in CATEGORICAL_FIELDS:
            value = request.form.get(field, "").strip()

            if not value:
                raise ValueError(f"Please enter {field}.")

            house_data[field] = value

        # Create DataFrame in the exact training feature order.
        input_data = pd.DataFrame(
            [house_data],
            columns=FEATURE_COLUMNS
        )

        # Predict house price.
        prediction = float(model.predict(input_data)[0])

        # Format prediction.
        predicted_price = f"${prediction:,.2f}"

        return render_template(
            "index.html",
            prediction=predicted_price,
            house_details=house_data
        )

    except (ValueError, TypeError, KeyError) as error:

        return render_template(
            "index.html",
            error=str(error),
            house_details=request.form
        )

    except Exception as error:

        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error="Prediction failed. Please check your inputs and try again.",
            house_details=request.form
        )


# --------------------------------------------------
# 5. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
