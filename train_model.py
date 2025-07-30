#!/usr/bin/env python3
"""
Model Training Script
This script trains the rainfall prediction model and saves it to disk.
"""

import sys
import os
from data_processing import load_and_clean_data, preprocess_data
from model import RainfallPredictor
from config import DATA_FILE, MODEL_FILE


def train_and_save_model(data_file=DATA_FILE, model_file=MODEL_FILE):
    """
    Train the model and save it to disk

    Args:
        data_file (str): Path to the CSV data file
        model_file (str): Path where to save the trained model
    """
    print("🚀 Starting model training...")

    try:
        # Load and clean data
        print("📊 Loading and cleaning data...")
        data = load_and_clean_data(data_file)
        print(f"✅ Data loaded successfully. Shape: {data.shape}")

        # Preprocess data
        print("🔧 Preprocessing data...")
        X, y, df_downsampled = preprocess_data(data)
        print(f"✅ Data preprocessed. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"   Class distribution: {y.value_counts().to_dict()}")

        # Train model
        print("🤖 Training Random Forest model with hyperparameter tuning...")
        predictor = RainfallPredictor()
        metrics = predictor.train_model(X, y, save_model=True, model_path=model_file)

        # Print results
        print("\n" + "=" * 50)
        print("🎯 TRAINING RESULTS")
        print("=" * 50)
        print(f"Best parameters: {metrics['best_params']}")
        print(f"Cross-validation scores: {metrics['cv_scores']}")
        print(f"Mean CV score: {metrics['mean_cv_score']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Model saved to: {model_file}")

        # Feature importance
        importance_df = predictor.get_feature_importance()
        print("\n📈 FEATURE IMPORTANCE")
        print("-" * 30)
        for _, row in importance_df.iterrows():
            print(f"{row['feature']:15s}: {row['importance']:.4f}")

        print("\n✅ Model training completed successfully!")
        return True

    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_file}' not found!")
        return False
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False


def main():
    """
    Main function
    """
    print("🌧️  Rainfall Prediction Model Trainer")
    print("=" * 40)

    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file '{DATA_FILE}' not found in current directory!")
        print(
            f"Please ensure the {DATA_FILE} file is in the same directory as this script."
        )
        sys.exit(1)

    # Train model
    success = train_and_save_model()

    if success:
        print("\n🎉 You can now run the Streamlit app with:")
        print("   streamlit run main.py")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
