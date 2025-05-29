# websocket_client.py
import websocket
import json
from src.data.preprocess import preprocess_imu_data
from src.features.extract_features import extract_features
from src.model.save_model import load_model
import numpy as np

# Window buffer
imu_window = []

def get_window():
    return imu_window

def start_websocket_client(ip_address="192.168.137.70"):
    ws = websocket.WebSocketApp(
        f"ws://{ip_address}:81/",
        on_message=on_message
    )
    ws.run_forever()

def on_message(ws, message):
    global imu_window
    data = json.loads(message)
    imu_window.append([
        data["accel_x"], data["accel_y"], data["accel_z"],
        data["gyro_x"], data["gyro_y"], data["gyro_z"]
    ])
    
    # Keep only the last N readings
    if len(imu_window) > 50:
        imu_window = imu_window[-50:]
