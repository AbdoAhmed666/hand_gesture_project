# src/data/preprocess_animation.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/gesture_lstm_model.h5")
scaler = joblib.load("models/scaler2.pkl")
label_encoder = joblib.load("models/label_encoder2.pkl")

def preprocess_and_predict_animation(window):
    window = np.array(window)
    X_scaled = scaler.transform(window.reshape(-1, 6)).reshape(1, window.shape[0], 6)
    probs = model.predict(X_scaled)[0]
    threshold = 0.7
    if np.max(probs) < threshold:
        return "Unknown Movement ❌"
    return label_encoder.inverse_transform([np.argmax(probs)])[0]
