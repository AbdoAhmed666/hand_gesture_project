import websocket
import threading
import json
import numpy as np
from queue import Queue

# Queue عشان نحط فيها القراءات
imu_data_queue = Queue()

def on_message(ws, message):
    """لما توصلك رسالة من الESP"""
    try:
        data = json.loads(message)
        reading = [
            data['accel_x'],
            data['accel_y'],
            data['accel_z'],
            data['gyro_x'],
            data['gyro_y'],
            data['gyro_z']
        ]
        imu_data_queue.put(reading)
    except Exception as e:
        print(f"Error parsing message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

def start_websocket_client(ip_address):
    """يشغل WebSocket client ويستمع للبيانات"""
    ws = websocket.WebSocketApp(
        f"ws://{ip_address}:81/",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()

def get_window(window_size=30):
    """ياخد نافذة readings كاملة"""
    window = []
    while len(window) < window_size:
        if not imu_data_queue.empty():
            window.append(imu_data_queue.get())
    return np.array(window)
