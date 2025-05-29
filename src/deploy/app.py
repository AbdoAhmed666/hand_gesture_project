# ✅ ملف app.py بعد التعديل الكامل لاستقبال IP تلقائيًا
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import os
import threading
import time
import sys
import logging

from src.utils.websocket_client import start_websocket_client, get_window
from src.data.unified_handler import unified_prediction_handler

# إعداد logger يدويًا مع دعم UTF-8
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.stream.reconfigure(encoding='utf-8')  # الدعم العربي هنا

logger.addHandler(stream_handler)

app = Flask(__name__, template_folder="templates", static_folder="static")
logger.info("✅ بدأ تشغيل تطبيق Flask...")

esp_ip = None  # 🧠 متغير لتخزين عنوان IP الخاص بـ ESP

# ✅ تشغيل WebSocket Client في thread منفصل بمجرد استقبال IP
def run_ws_client():
    global esp_ip
    while esp_ip is None:
        logger.info(" في انتظار استقبال عنوان IP من ESP...")
        time.sleep(2)
    start_websocket_client(esp_ip)


# 🧠 تخزين آخر نتيجة تنبؤ
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

# ✅ مسار جديد لتسجيل IP المُرسل من ESP8266
@app.route("/register_ip", methods=["POST"])
def register_ip():
    global esp_ip
    try:
        esp_ip = request.json.get("ip")
        logger.info(f" تم تسجيل عنوان IP الخاص بـ ESP: {esp_ip}")

        ws_thread = threading.Thread(target=run_ws_client)
        ws_thread.daemon = True
        ws_thread.start()        
        
        return jsonify({"status": "received", "ip": esp_ip})
    except Exception as e:
        logger.error(f"❌ خطأ في تسجيل IP: {e}")
        return jsonify({"error": "Invalid request"}), 400
