import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, save_model, evaluate_model
from src.prediction_interface import predict_iem
from src.evaluation import plot_confusion_matrix  # assuming this exists

# Set your dataset path here
DATA_PATH = os.path.join("data", "processed", "iem_processed.csv")
MODEL_PATH = "saved_models/random_forest_model.pkl"

def main():
    # Step 1: Load raw data
    df = load_data(DATA_PATH)
    if df is None:
        return

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df, label_column='diagnosis')

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Save model
    save_model(model, MODEL_PATH)

    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 6: Plot confusion matrix
    class_names = sorted(df['diagnosis'].unique())  # Adjust if you're encoding labels
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, class_names=class_names)

    # Step 7 (Optional): Test prediction
    sample_input = X_test.iloc[0]  # Use .iloc for DataFrame row access
    print(f"\n[INFO] Sample prediction: {predict_iem(sample_input)}")


if __name__ == "__main__":
    main()
