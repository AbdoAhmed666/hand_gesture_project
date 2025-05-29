# test_model.py
import joblib
import tensorflow as tf

if __name__ == "__main__":
    # 1. حمّل الموديل
    try:
        model = joblib.load("models/gesture_model.pkl")
        print("✅ موديل الـ joblib اتحمّل من غير مشاكل")
    except Exception as e:
        print("❌ خطأ في تحميل الموديل:", e)
        exit(1)

    # 2. لو الموديل فعلاً Keras model جرب تطبع الـ summary
    try:
        # tf.keras.models.Model عندنا لازم يكون ده النوع
        if isinstance(model, tf.keras.Model):
            model.summary()
        else:
            print(f"🔔 الموديل مش من نوع tf.keras.Model، ده نوعه: {type(model)}")
    except Exception as e:
        print("❌ خطأ وإحنا بنطبع الـ summary:", e)
