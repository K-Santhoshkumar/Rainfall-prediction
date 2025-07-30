"""
Configuration file for the Rainfall Prediction System
Contains all the settings, parameters, and file paths used throughout the application
"""

import os

# File paths
DATA_FILE = "Rainfall.csv"
MODEL_FILE = "rainfall_prediction_model.pkl"

# Data processing parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Cross-validation parameters
CV_FOLDS = 5

# Features to drop (highly correlated)
FEATURES_TO_DROP = ["maxtemp", "temparature", "mintemp"]

# EDA features for visualization
EDA_FEATURES = [
    "pressure",
    "maxtemp",
    "temparature",
    "mintemp",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "windspeed",
]

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Rainfall Prediction App",
    "page_icon": "🌧️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Sample input data for testing
SAMPLE_INPUT_DATA = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

# Feature descriptions for UI
FEATURE_DESCRIPTIONS = {
    "pressure": "Atmospheric pressure in hPa",
    "dewpoint": "Dew point temperature in °C",
    "humidity": "Relative humidity in %",
    "cloud": "Cloud cover in %",
    "sunshine": "Sunshine hours",
    "winddirection": "Wind direction in degrees",
    "windspeed": "Wind speed in km/h",
}

# Validation ranges for input features
FEATURE_RANGES = {
    "pressure": (900, 1100),
    "dewpoint": (-50, 50),
    "humidity": (0, 100),
    "cloud": (0, 100),
    "sunshine": (0, 24),
    "winddirection": (0, 360),
    "windspeed": (0, 200),
}

# UI styling
CSS_STYLES = """
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-result {
    font-size: 1.5rem;
    font-weight: bold;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.rainfall {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.no-rainfall {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
"""

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# Model evaluation metrics
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1_score"]


def get_data_path():
    """Get the full path to the data file"""
    return os.path.join(os.getcwd(), DATA_FILE)


def get_model_path():
    """Get the full path to the model file"""
    return os.path.join(os.getcwd(), MODEL_FILE)


def validate_input_data(input_data):
    """
    Validate input data against defined ranges

    Args:
        input_data (dict): Dictionary of feature names and values

    Returns:
        tuple: (is_valid, error_message)
    """
    for feature, value in input_data.items():
        if feature in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= value <= max_val):
                return (
                    False,
                    f"{feature} value {value} is outside valid range [{min_val}, {max_val}]",
                )

    return True, ""
