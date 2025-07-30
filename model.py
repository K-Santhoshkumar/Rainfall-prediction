import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import RANDOM_FOREST_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_STATE


class RainfallPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def train_model(
        self, X, y, save_model=True, model_path="rainfall_prediction_model.pkl"
    ):
        """
        Train the Random Forest model with hyperparameter tuning
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Initialize Random Forest model
        rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

        # Define parameter grid for hyperparameter tuning
        param_grid_rf = RANDOM_FOREST_PARAMS

        # Perform Grid Search
        grid_search_rf = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid_rf,
            cv=CV_FOLDS,
            n_jobs=-1,
            verbose=0,
        )
        grid_search_rf.fit(X_train, y_train)

        # Get best model
        self.model = grid_search_rf.best_estimator_
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        # Evaluate model
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=CV_FOLDS)
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            "best_params": grid_search_rf.best_params_,
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        }

        # Save model if requested
        if save_model:
            self.save_model(model_path)

        return metrics

    def predict(self, input_data):
        """
        Make prediction on new data
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")

        # Convert input to DataFrame if it's not already
        if isinstance(input_data, (list, tuple)):
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input data must be a list, tuple, or pandas DataFrame")

        # Make prediction
        prediction = self.model.predict(input_df)[0]
        result = "Rainfall" if prediction == 1 else "No Rainfall"

        return {
            "prediction": prediction,
            "result": result,
            "probability": (
                self.model.predict_proba(input_df)[0]
                if hasattr(self.model, "predict_proba")
                else None
            ),
        }

    def save_model(self, file_path="rainfall_prediction_model.pkl"):
        """
        Save the trained model and feature names
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")

        model_data = {"model": self.model, "feature_names": self.feature_names}

        with open(file_path, "wb") as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path="rainfall_prediction_model.pkl"):
        """
        Load a trained model from file
        """
        with open(file_path, "rb") as file:
            model_data = pickle.load(file)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = True

    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df


def load_trained_model(file_path="rainfall_prediction_model.pkl"):
    """
    Load a trained model from file (convenience function)
    """
    predictor = RainfallPredictor()
    predictor.load_model(file_path)
    return predictor
