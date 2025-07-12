import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# --- Sidebar Configuration ---
st.sidebar.title("🏭 MMS Safety System")
st.sidebar.markdown("**Configuration Panel**")

input_source = st.sidebar.radio(
    "📹 Input Source",
    ["Webcam", "Image Upload", "Video Upload"],
    index=0,
    help="Select the source for hazard detection monitoring"
)

hazard_threshold = st.sidebar.slider(
    "🎯 Hazard Detection Sensitivity",
    5000, 50000, 15000, 1000,
    help="Higher values = more sensitive detection (may increase false alarms)"
)

confidence_threshold = st.sidebar.slider(
    "⚡ Detection Confidence",
    0.1, 1.0, 0.8, 0.1,
    help="Higher values = stricter detection (fewer false positives)"
)

# --- Session State Management ---
if 'hazard_count' not in st.session_state:
    st.session_state['hazard_count'] = 0
if 'machine_status' not in st.session_state:
    st.session_state['machine_status'] = "Safe"

# --- Main Header ---
st.markdown("""
<div style="text-align:center; margin-bottom:2rem;">
    <h1>🏭 MMS Safety System</h1>
    <p>AI-Powered Accident Prevention in Modular Manufacturing Systems</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
MODEL_PATH = 'kaggle_models/construction-safety/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt'

# Check if model file exists, otherwise use a default model
import os
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    # Use a smaller default model for deployment
    st.warning("⚠️ Custom model not found. Using default YOLOv8n model.")
    model = YOLO('yolov8n.pt')  # This will download automatically

# --- Main UI Logic ---
if input_source == "Webcam":
    st.markdown('<div style="background:#0e1117;padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;">📹 Webcam Feed</div>', unsafe_allow_html=True)
    run = st.button("🚀 Start Real-Time Monitoring", type="primary")
    stop = st.button("Stop Webcam", key="stop_webcam_button")
    status_placeholder = st.empty()
    metric_col1, metric_col2 = st.columns(2)
    webcam_placeholder = st.empty()
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    if run:
        st.session_state['webcam_running'] = True

    if stop:
        st.session_state['webcam_running'] = False

    if st.session_state['webcam_running']:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while cap.isOpened() and st.session_state['webcam_running']:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ Failed to read frame from webcam. Retrying...")
                time.sleep(1)
                continue
            results = model(frame, conf=confidence_threshold)
            first_result = None
            if results is not None and hasattr(results, '__len__') and len(results) > 0:
                try:
                    first_result = results[0]
                except Exception:
                    first_result = None
            hazard_detected = False
            if first_result is not None:
                for box in first_result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id] if model.names and class_id in model.names else str(class_id)
                    if label == "person" and box.conf[0] > confidence_threshold:
                        hazard_detected = True
            if hazard_detected:
                st.session_state['hazard_count'] += 1
                st.session_state['machine_status'] = "STOPPED"
            else:
                st.session_state['machine_status'] = "Safe"
            annotated_img = first_result.plot() if first_result is not None else frame
            webcam_placeholder.image(annotated_img, channels="BGR")
            # Status Card
            status_html = f'<div style="background:#232323;padding:1.5rem;border-radius:12px;border-left:5px solid {"#ef4444" if hazard_detected else "#10b981"};margin-bottom:1rem;color:inherit;"><h3>{"🚨 EMERGENCY STOP" if hazard_detected else "✅ SYSTEM SAFE"}</h3><p>{"Hazard detected in machine zone" if hazard_detected else "No hazards detected"}</p></div>'
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            # Metric Cards
            # REMOVE the metric_col1 and metric_col2 blocks from here

        cap.release()
        st.success("Webcam stopped.")

        # Show metrics ONCE after the loop
        with metric_col1:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{st.session_state["hazard_count"]}</div><div style="font-size:0.9rem;opacity:0.7;">Hazard Events</div></div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{st.session_state["machine_status"]}</div><div style="font-size:0.9rem;opacity:0.7;">Machine Status</div></div>', unsafe_allow_html=True)

elif input_source == "Image Upload":
    st.markdown('<div style="background:#0e1117;padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;">🖼️ Image Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    status_placeholder = st.empty()
    metric_col1, metric_col2 = st.columns(2)
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        results = model(img_array, conf=confidence_threshold)
        first_result = None
        if results is not None and hasattr(results, '__len__') and len(results) > 0:
            try:
                first_result = results[0]
            except Exception:
                first_result = None
        hazard_detected = False
        if first_result is not None:
            for box in first_result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id] if model.names and class_id in model.names else str(class_id)
                if label == "person" and box.conf[0] > confidence_threshold:
                    hazard_detected = True
        if hazard_detected:
            st.session_state['hazard_count'] += 1
            st.session_state['machine_status'] = "STOPPED"
        else:
            st.session_state['machine_status'] = "Safe"
        annotated_img = first_result.plot() if first_result is not None else img_array
        st.image(annotated_img, caption="Detection Result", use_column_width=True)
        # Status Card
        status_html = f'<div style="background:#232323;padding:1.5rem;border-radius:12px;border-left:5px solid {"#ef4444" if hazard_detected else "#10b981"};margin-bottom:1rem;color:inherit;"><h3>{"🚨 EMERGENCY STOP" if hazard_detected else "✅ SYSTEM SAFE"}</h3><p>{"Hazard detected in machine zone" if hazard_detected else "No hazards detected"}</p></div>'
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        # Metric Cards
        with metric_col1:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{st.session_state["hazard_count"]}</div><div style="font-size:0.9rem;opacity:0.7;">Hazard Events</div></div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{st.session_state["machine_status"]}</div><div style="font-size:0.9rem;opacity:0.7;">Machine Status</div></div>', unsafe_allow_html=True)

elif input_source == "Video Upload":
    st.markdown('<div style="background:#0e1117;padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;">🎬 Video Upload</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    status_placeholder = st.empty()
    metric_col1, metric_col2 = st.columns(2)
    progress_bar = st.progress(0)
    if uploaded_video is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        hazard_count_video = 0
        machine_status_video = "Safe"
        first_frame_annotated = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            results = model(frame, conf=confidence_threshold)
            first_result = None
            if results is not None and hasattr(results, '__len__') and len(results) > 0:
                try:
                    first_result = results[0]
                except Exception:
                    first_result = None
            hazard_detected = False
            if first_result is not None:
                for box in first_result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id] if model.names and class_id in model.names else str(class_id)
                    if label == "person" and box.conf[0] > confidence_threshold:
                        hazard_detected = True
            if hazard_detected:
                hazard_count_video += 1
                machine_status_video = "STOPPED"
            else:
                machine_status_video = "Safe"
            if first_frame_annotated is None and first_result is not None:
                first_frame_annotated = first_result.plot()
            progress_bar.progress(frame_idx / total_frames)
        cap.release()
        if first_frame_annotated is not None:
            st.image(first_frame_annotated, caption="First Frame Detection Result", use_column_width=True)
        # Status Card
        status_html = f'<div style="background:#232323;padding:1.5rem;border-radius:12px;border-left:5px solid {"#ef4444" if hazard_count_video > 0 else "#10b981"};margin-bottom:1rem;color:inherit;"><h3>{"🚨 EMERGENCY STOP" if hazard_count_video > 0 else "✅ SYSTEM SAFE"}</h3><p>{"Hazard detected in machine zone" if hazard_count_video > 0 else "No hazards detected"}</p></div>'
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        # Metric Cards
        with metric_col1:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{hazard_count_video}</div><div style="font-size:0.9rem;opacity:0.7;">Hazard Events (Video)</div></div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f'<div style="background:#232323;padding:1.5rem;border-radius:12px;text-align:center;color:inherit;"><div style="font-size:2rem;font-weight:700;">{machine_status_video}</div><div style="font-size:0.9rem;opacity:0.7;">Machine Status</div></div>', unsafe_allow_html=True)
        progress_bar.empty()



