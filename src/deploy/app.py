# ✅ ملف app.py بعد التعديل الكامل
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import os
import threading
from src.utils.websocket_client import start_websocket_client, get_window
from src.utils.logger import get_logger
from src.data.unified_handler import unified_prediction_handler

app = Flask(__name__, template_folder="templates", static_folder="static")
logger = get_logger(__name__)
logger.info("Started running Flask App...")

# تشغيل WebSocket Client في thread منفصل
def run_ws_client():
    start_websocket_client("192.168.137.60")  # ✅ غيّر الـ IP حسب جهازك

ws_thread = threading.Thread(target=run_ws_client)
ws_thread.daemon = True
ws_thread.start()

# تخزين آخر نتيجة تنبؤ
temp_prediction = {"gesture": "لا يوجد بعد", "data_type": "غير معروف"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global temp_prediction
    try:
        imu_data = get_window()

        if len(imu_data) < 1:
            return jsonify({"error": "❌ لا توجد بيانات كافية من IMU بعد"}), 400

        prediction_result = unified_prediction_handler(imu_data)

        if isinstance(prediction_result, dict) and "error" in prediction_result:
            return jsonify(prediction_result), 400

        gesture = prediction_result["gesture"]
        data_type = prediction_result["data_type"]

        temp_prediction = {"gesture": gesture, "data_type": data_type}

        logger.info(f"✅ Prediction: {gesture} | Type: {data_type}")
        return jsonify(temp_prediction)

    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/latest_prediction", methods=["GET"])
def get_latest_prediction():
    return jsonify(temp_prediction)

@app.route('/dashboard')
def dashboard():
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'index.html')
