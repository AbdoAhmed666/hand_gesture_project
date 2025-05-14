# src/data/unified_handler.py
from src.data.preprocess_fixed import preprocess_and_predict_fixed
from src.data.preprocess_animation import preprocess_and_predict_animation
import numpy as np

def determine_data_type(data):
    window = np.array(data)
    diff = np.abs(np.diff(window, axis=0))
    mean_diff = np.mean(diff)

    if mean_diff < 0.2:
        return "fixed"
    else:
        return "animation"

def unified_prediction_handler(raw_data):
    data_type = determine_data_type(raw_data)
    
    if data_type == "fixed":
        gesture = preprocess_and_predict_fixed(raw_data)
    elif data_type == "animation":
        gesture = preprocess_and_predict_animation(raw_data)
    else:
        return {"error": "❌ نوع البيانات غير معروف"}
    
    return {"gesture": gesture, "data_type": data_type}
