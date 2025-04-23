# src/__init__.py

# This makes sure the submodules can be imported easily.
from .data_preprocessing import preprocess_data, load_data
from .model_training import train_model, save_model, load_model
from .evaluation import evaluate_model, plot_confusion_matrix
from .prediction_interface import predict_iem

__all__ = [
    "preprocess_data",
    "load_data",
    "train_model",
    "save_model",
    "load_model",
    "evaluate_model",
    "plot_confusion_matrix",
    "predict_iem"
]
