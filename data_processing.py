import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from config import EDA_FEATURES, FEATURES_TO_DROP, RANDOM_STATE


def load_and_clean_data(file_path="Rainfall.csv"):
    """
    Load and clean the rainfall dataset
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Remove extra spaces in column names
    data.columns = data.columns.str.strip()

    # Drop 'day' column as it's not needed for prediction
    data = data.drop(columns=["day"])

    # Handle missing values
    data["winddirection"] = data["winddirection"].fillna(
        data["winddirection"].mode()[0]
    )
    data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

    # Convert rainfall to binary (1/0)
    data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

    return data


def perform_eda(data):
    """
    Perform Exploratory Data Analysis
    """
    # Set plot style
    sns.set(style="whitegrid")

    # Create histograms
    fig_hist, axes_hist = plt.subplots(3, 3, figsize=(15, 10))
    for i, feature in enumerate(EDA_FEATURES):
        ax = axes_hist[i // 3, i % 3]
        sns.histplot(data[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
    plt.tight_layout()

    # Create boxplots
    fig_box, axes_box = plt.subplots(3, 3, figsize=(15, 10))
    for i, feature in enumerate(EDA_FEATURES):
        ax = axes_box[i // 3, i % 3]
        sns.boxplot(y=data[feature], ax=ax)
        ax.set_title(f"Boxplot of {feature}")
    plt.tight_layout()

    # Rainfall distribution
    fig_rainfall, ax_rainfall = plt.subplots(figsize=(6, 4))
    sns.countplot(x="rainfall", data=data, ax=ax_rainfall)
    ax_rainfall.set_title("Distribution of Rainfall")

    # Correlation heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Correlation heatmap")

    return {
        "histograms": fig_hist,
        "boxplots": fig_box,
        "rainfall_dist": fig_rainfall,
        "correlation": fig_corr,
    }


def preprocess_data(data):
    """
    Preprocess data for model training
    """
    # Drop highly correlated columns
    data = data.drop(columns=FEATURES_TO_DROP)

    # Separate majority and minority classes
    df_majority = data[data["rainfall"] == 1]
    df_minority = data[data["rainfall"] == 0]

    # Downsample majority class to match minority count
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=RANDOM_STATE,
    )

    # Combine and shuffle
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # Split features and target
    X = df_downsampled.drop(columns=["rainfall"])
    y = df_downsampled["rainfall"]

    return X, y, df_downsampled


def get_data_info(data):
    """
    Get basic information about the dataset
    """
    info = {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "missing_values": data.isnull().sum().to_dict(),
        "rainfall_counts": data["rainfall"].value_counts().to_dict(),
        "head": data.head(),
        "tail": data.tail(),
    }
    return info
