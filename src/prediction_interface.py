import numpy as np
from sklearn.preprocessing import StandardScaler
from .model_training import load_model

# Keep the scaler consistent with training
scaler = StandardScaler()

def predict_iem(input_data, model_path='saved_models/random_forest_model.pkl'):
    """
    Predicts IEM class for a new patient based on input features.

    Args:
        input_data: List or array of patient features (raw)
        model_path: Path to the saved model file

    Returns:
        Predicted IEM label
    """
    # Convert input to 2D array (if user gives 1D list)
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input the same way as training (assumes scaler fitted on training set)
    # NOTE: In a real pipeline, you should save the scaler too. Here we'll assume standardized inputs.
    scaled_input = scaler.fit_transform(input_array)  # For demo. In real use: load scaler.

    # Load model and predict
    model = load_model(model_path)
    prediction = model.predict(scaled_input)

    return prediction[0]
