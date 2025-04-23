import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest classifier on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Seed for reproducibility
        
    Returns:
        Trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints classification metrics.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    print("🔍 Model Evaluation:\n", classification_report(y_test, y_pred))

def save_model(model, filepath='saved_models/random_forest_model.pkl'):
    """
    Saves the trained model to the specified filepath.
    
    Args:
        model: Trained model
        filepath: Destination path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Saved model to {filepath}")

def load_model(filepath='saved_models/random_forest_model.pkl'):
    """
    Loads a trained model from the specified filepath.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] No model found at {filepath}")
    model = joblib.load(filepath)
    print(f"[INFO] Loaded model from {filepath}")
    return model
