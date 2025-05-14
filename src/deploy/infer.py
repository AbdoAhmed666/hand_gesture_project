from src.model.save_load import load_model
from src.features.extract_features import flatten_window
import numpy as np

def predict(window, model):
    features = flatten_window(window).reshape(1, -1)
    return model.predict(features)[0]