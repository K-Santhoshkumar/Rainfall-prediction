import streamlit as st
import pandas as pd
import numpy as np
import io
from data_processing import (
    load_and_clean_data,
    perform_eda,
    preprocess_data,
    get_data_info,
)
from model import RainfallPredictor, load_trained_model
import matplotlib.pyplot as plt


def render_prediction_page():
    """
    Render the rainfall prediction page
    """
    st.title("Rainfall Prediction")
    st.write("Enter the following weather parameters:")

    try:
        # Load the trained model
        predictor = load_trained_model()
        feature_names = predictor.feature_names

        # Input fields for each feature
        input_values = []
        input_valid = True

        for feature in feature_names:
            value = st.text_input(f"{feature.capitalize()}")
            if value == "":
                input_valid = False
            input_values.append(value)

        if st.button("Predict"):
            if not input_valid:
                st.warning("Please fill in all input fields.")
            else:
                # Convert input values to float
                try:
                    input_floats = [float(v) for v in input_values]
                    prediction_result = predictor.predict(input_floats)

                    # Display result
                    st.success(f"Prediction result: {prediction_result['result']}")

                    # Display probability if available
                    if prediction_result["probability"] is not None:
                        prob_rainfall = prediction_result["probability"][1]
                        prob_no_rainfall = prediction_result["probability"][0]

                        st.write("Prediction Probabilities:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rainfall", f"{prob_rainfall:.2%}")
                        with col2:
                            st.metric("No Rainfall", f"{prob_no_rainfall:.2%}")

                except ValueError:
                    st.error("Please enter valid numbers for all fields.")

    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")


def render_analysis_page():
    """
    Render the data analysis and visualization page
    """
    st.title("Rainfall Prediction - Full Analysis")

    try:
        # Load and process data
        data = load_and_clean_data()
        data_info = get_data_info(data)

        # 1. Data Collection and Processing
        st.subheader("Data Head")
        st.dataframe(data_info["head"])

        st.subheader("Data Tail")
        st.dataframe(data_info["tail"])

        st.subheader("Data Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Missing Values (before cleaning)")
        st.write(data_info["missing_values"])

        st.subheader("Rainfall Value Counts (before balancing)")
        st.write(data_info["rainfall_counts"])

        # 2. EDA
        st.subheader("Feature Distributions (Histograms)")
        eda_plots = perform_eda(data)
        st.pyplot(eda_plots["histograms"])

        st.subheader("Feature Distributions (Boxplots)")
        st.pyplot(eda_plots["boxplots"])

        st.subheader("Rainfall Distribution")
        st.pyplot(eda_plots["rainfall_dist"])

        st.subheader("Correlation Heatmap")
        st.pyplot(eda_plots["correlation"])

        # 3. Data Preprocessing
        st.subheader("Drop Highly Correlated Columns")
        X, y, df_downsampled = preprocess_data(data)
        st.write("Columns after dropping:", X.columns.tolist())

        st.subheader("Rainfall Value Counts (after dropping columns)")
        st.write(data["rainfall"].value_counts())

        # 4. Class Balancing
        df_majority = data[data["rainfall"] == 1]
        df_minority = data[data["rainfall"] == 0]
        st.write("Majority class shape:", df_majority.shape)
        st.write("Minority class shape:", df_minority.shape)

        st.subheader("Class Balance After Downsampling")
        st.write(df_downsampled["rainfall"].value_counts())
        st.write("Downsampled Data Head:")
        st.dataframe(df_downsampled.head())

        # 5. Model Training and Evaluation
        st.subheader("Model Training and Evaluation")

        # Train model
        predictor = RainfallPredictor()
        metrics = predictor.train_model(
            X, y, save_model=False
        )  # Don't save during analysis

        st.write("Best parameters for Random Forest:", metrics["best_params"])
        st.write("Cross-validation scores:", metrics["cv_scores"])
        st.write("Mean cross-validation score:", metrics["mean_cv_score"])
        st.write("Test set Accuracy:", metrics["test_accuracy"])
        st.write("Test set Confusion Matrix:")
        st.write(metrics["confusion_matrix"])
        st.write("Classification Report:")
        st.text(metrics["classification_report"])

        # Feature importance
        st.subheader("Feature Importance")
        importance_df = predictor.get_feature_importance()
        st.dataframe(importance_df)

        # 6. Sample Prediction
        st.subheader("Sample Prediction")
        input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
        prediction_result = predictor.predict(input_data)
        st.write(f"Input: {input_data}")
        st.write(f"Prediction result: {prediction_result['result']}")

    except FileNotFoundError:
        st.error(
            "Data file not found. Please ensure 'Rainfall.csv' is in the current directory."
        )
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")


def main_ui():
    """
    Main UI function that sets up the Streamlit interface
    """
    # Sidebar navigation
    st.sidebar.title("Rainfall Prediction App")
    page = st.sidebar.radio("Go to", ["Predict Rainfall", "Visualizations & Outputs"])

    # Render appropriate page
    if page == "Predict Rainfall":
        render_prediction_page()
    elif page == "Visualizations & Outputs":
        render_analysis_page()


if __name__ == "__main__":
    main_ui()
