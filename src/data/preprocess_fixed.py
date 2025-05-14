# src/data/preprocess_fixed.py
import joblib
from src.features.extract_features import extract_features
import numpy as np

# تحميل النموذج والسكيلر والمحّول
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler1.pkl")
label_encoder = joblib.load("models/label_encoder1.pkl")

def preprocess_and_predict_fixed(window):
    """
    - window: مصفوفة numpy بحجم (عدد العينات, 6 خصائص مثل acc و gyro)
    """
    # استخراج الخصائص (features)
    feature_vector = extract_features(np.array(window))  # تأكد إنها numpy

    # إعادة تشكيلها لتكون بشكل (1, N)
    feature_vector = feature_vector.reshape(1, -1)

    # تطبيع البيانات
    X_scaled = scaler.transform(feature_vector)

    # التنبؤ بالاحتمالات
    probs = model.predict_proba(X_scaled)[0]
    threshold = 0.7
    if max(probs) < threshold:
        return "Unknown Movement ❌"

    # التنبؤ بالفئة وإرجاع الاسم المقابل
    prediction = model.predict(X_scaled)[0]
    return label_encoder.inverse_transform([prediction])[0]
