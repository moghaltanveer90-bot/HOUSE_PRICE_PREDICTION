
"""
Project Title: End-to-End House Price Prediction
Author: Moghal Tanveer

Description:
Loads and cleans the house-price dataset, preprocesses numerical and
categorical features, trains an XGBoost regression model, evaluates it,
saves the complete preprocessing/model pipeline, and predicts new prices.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


class HousePricePrediction:
    """Prepare data, train, evaluate, save and predict house prices."""

    def __init__(self, data_path):
        """Load the dataset and prepare features and target."""

        self.data = pd.read_csv(data_path)

        if "price" not in self.data.columns:
            raise ValueError("Dataset must contain a 'price' column.")

        # Remove unnecessary columns.
        self.data = self.data.drop(
            columns=["date", "street"],
            errors="ignore"
        )

        # Clean the target variable.
        self.data["price"] = pd.to_numeric(
            self.data["price"], errors="coerce"
        )

        self.data = self.data.dropna(subset=["price"])
        self.data = self.data[self.data["price"] > 0].copy()

        if self.data.empty:
            raise ValueError("No valid house prices found.")

        # Separate features and target.
        self.X = self.data.drop(columns=["price"])
        self.y = self.data["price"]

        if self.X.shape[1] == 0:
            raise ValueError("No usable feature columns found.")

        # Identify numerical and categorical columns.
        self.numeric_columns = self.X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        self.categorical_columns = self.X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        # Split data into training and testing sets.
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=42
        )

        # Numerical preprocessing.
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        # Categorical preprocessing.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine preprocessing steps.
        transformers = []

        if self.numeric_columns:
            transformers.append(
                ("numeric", numeric_pipeline, self.numeric_columns)
            )

        if self.categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline,
                 self.categorical_columns)
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers
        )

        # Create the XGBoost machine learning pipeline.
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),

                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=1500,
                        learning_rate=0.02,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        reg_lambda=3,
                        reg_alpha=0.05,
                        gamma=0,
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist"
                    )
                )
            ]
        )

        # Transform house prices using log1p.
        self.model = TransformedTargetRegressor(
            regressor=xgb_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        print("\nDATASET INFORMATION")
        print("-------------------")
        print("Dataset shape:", self.data.shape)
        print("Training records:", len(self.X_train))
        print("Testing records:", len(self.X_test))
        print("Numerical features:", self.numeric_columns)
        print("Categorical features:", self.categorical_columns)
        print("Removed columns: date, street (if present)")

    def train_model(self):
        """Train the complete preprocessing and XGBoost pipeline."""

        self.model.fit(self.X_train, self.y_train)

        print("\nModel training completed successfully!")

        return self.model

    def test_model(self):
        """Evaluate the model using unseen testing data."""

        predictions = self.model.predict(self.X_test)

        mae = mean_absolute_error(
            self.y_test, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(self.y_test, predictions)
        )

        r2 = r2_score(
            self.y_test, predictions
        )

        print("\nMODEL TESTING RESULTS")
        print("---------------------")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"R2 Percentage: {r2 * 100:.2f}%")

        print("\nNote: R2 is a regression evaluation metric.")
        print("R2 percentage is not classification accuracy.")

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }

    def save_model(self, model_path="models/house_price_model.pkl"):
        """Save the complete trained pipeline using Joblib."""

        model_directory = os.path.dirname(model_path)

        os.makedirs(model_directory or ".", exist_ok=True)

        joblib.dump(self.model, model_path)

        print("\nModel saved successfully!")
        print("Model location:", model_path)

    def predict_price(self, input_data):
        """Predict the price of a new house."""

        input_df = pd.DataFrame(
            [input_data]
        ).reindex(columns=self.X.columns)

        prediction = self.model.predict(input_df)

        return float(prediction[0])


# Main program execution.
if __name__ == "__main__":

    # Create the prediction object.
    house = HousePricePrediction("data.csv")

    # Train the model.
    house.train_model()

    # Evaluate the model.
    metrics = house.test_model()

    # Save the trained model.
    house.save_model()

    # Test prediction using one testing record.
    sample_house = house.X_test.iloc[0].to_dict()

    predicted_price = house.predict_price(sample_house)
    actual_price = float(house.y_test.iloc[0])

    print("\nSAMPLE HOUSE PREDICTION")
    print("-----------------------")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Actual Price: ${actual_price:,.2f}")

    print("\nPROJECT EXECUTED SUCCESSFULLY!")
