from src import (
    load_data,
    preprocess_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
    plot_confusion_matrix,
    predict_iem
)

def run_training():
    df = load_data("data/raw/iem_data.csv")
    if df is None:
        return

    # Step 1: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 2: Train the model
    model = train_model(X_train, y_train)

    # Step 3: Save the trained model
    save_model(model)
    print("[DONE] Training and saving complete.")

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)  # Add model evaluation here


def run_evaluation():
    df = load_data("data/raw/iem_data.csv")
    if df is None:
        return

    # Step 1: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 2: Load the trained model
    model = load_model()

    # Step 3: Evaluate the model
    y_true, y_pred = evaluate_model(model, X_test, y_test)

    # Step 4: Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred)


def run_prediction():
    # Sample patient feature vector (you’ll replace this with real data or test input)
    sample_data = [0.45, 88.1, 5.4, 0.02, 12.3, 55.1, 0.1, 1.5]  # Example placeholder
    prediction = predict_iem(sample_data)
    print(f"🧬 Prediction for new sample: {prediction}")


if __name__ == "__main__":
    print("=== IEM ML Project Runner ===\n")

    # === Choose what you want to run ===
    
    # Uncomment the ones you want:
    # run_training()
    # run_evaluation()
    # run_prediction()

    print("\n✅ Task finished.")
