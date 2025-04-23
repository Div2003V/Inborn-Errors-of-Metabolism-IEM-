import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, save_model, load_model
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.prediction_interface import predict_iem
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/iem_processed.csv"
MODEL_PATH = "saved_models/random_forest_model.pkl"

st.title("🧬 Inborn Errors of Metabolism (IEM) - ML Detection App")

df = load_data(DATA_PATH)

if df is not None:
    st.success(f"✅ Data loaded successfully. Shape: {df.shape}")

    if st.button("🚀 Train & Evaluate Model"):
        X_train, X_test, y_train, y_test = preprocess_data(df, label_column="diagnosis")
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH)
        y_true, y_pred = evaluate_model(model, X_test, y_test)

        st.subheader("📋 Evaluation Report")
        st.text(f"True Labels: {y_true}\nPredicted Labels: {y_pred}")

        st.subheader("📊 Confusion Matrix")
        fig = plot_confusion_matrix(y_true, y_pred, class_names=sorted(df['diagnosis'].unique()))
        st.pyplot(fig)

    if st.button("🔮 Predict Sample"):
        # This should match the number of features your model was trained on
        sample_input = [0.45, 88.1, 5.4, 0.02, 12.3, 55.1, 0.1]  # adjust to fit your actual features
        prediction = predict_iem(sample_input)
        st.success(f"Predicted Class: {prediction}")

