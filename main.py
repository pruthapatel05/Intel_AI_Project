import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import threading
import io
from PIL import Image

app = Flask(__name__)

# Global state for hazard detection
auto_stop_triggered = False
hazard_event_count = 0
hazard_sensitivity = 5000
last_frame = None
lock = threading.Lock()

# Placeholder AI hazard detection (red object detection)
def detect_hazard(frame, threshold):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    red_pixels = cv2.countNonZero(mask)
    hazard = red_pixels > threshold
    return hazard, red_pixels

def gen_frames():
    global auto_stop_triggered, hazard_event_count, last_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            threshold = hazard_sensitivity
        hazard, red_pixels = detect_hazard(frame, threshold)
        with lock:
            auto_stop_triggered = hazard
            if hazard:
                hazard_event_count += 1
            last_frame = frame.copy()
        # No rectangle here; overlays will be done in JS/HTML
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('interactive.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/machine_status')
def machine_status():
    with lock:
        stopped = auto_stop_triggered
        count = hazard_event_count
    return jsonify({"auto_stop": stopped, "hazard_count": count})

@app.route('/set_sensitivity', methods=['POST'])
def set_sensitivity():
    global hazard_sensitivity
    data = request.get_json()
    value = int(data.get('sensitivity', 5000))
    with lock:
        hazard_sensitivity = value
    return jsonify({"success": True, "sensitivity": hazard_sensitivity})

@app.route('/snapshot')
def snapshot():
    with lock:
        frame = last_frame.copy() if last_frame is not None else None
    if frame is None:
        return "No frame available", 404
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name='snapshot.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 