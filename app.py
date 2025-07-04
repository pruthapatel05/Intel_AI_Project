import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

st.set_page_config(
    page_title="AI-Based Accident Prevention in MMS",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("Configuration")
input_source = st.sidebar.radio("Input Source", ["Webcam", "Image Upload"], index=0)
hazard_threshold = st.sidebar.slider("Hazard Detection Sensitivity", 1000, 20000, 5000, 500)
st.sidebar.markdown("---")
st.sidebar.info("This app uses AI to detect hazards in real-time and simulate machine auto-stop.")

# --- State ---
if 'hazard_count' not in st.session_state:
    st.session_state['hazard_count'] = 0
if 'last_snapshot' not in st.session_state:
    st.session_state['last_snapshot'] = None

# --- Main Title ---
st.title("AI-Based Accident Prevention in MMS")
st.markdown("""
**Organisation:** International Automobile Centre of Excellence, Ahmedabad  
**Category:** Industry Defined Problem
""")

# --- Hazard Detection Function ---
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
    return hazard, mask, red_pixels

# --- Main Area ---
status_circle_placeholder = st.empty()
status_placeholder = st.empty()
hazard_count_placeholder = st.empty()
snapshot_placeholder = st.empty()

# --- Machine ROI (fixed rectangle) ---
def draw_machine_highlight(frame, hazard):
    h, w = frame.shape[:2]
    # Define ROI (centered rectangle, 40% width, 40% height)
    rw, rh = int(w * 0.4), int(h * 0.4)
    x1, y1 = w // 2 - rw // 2, h // 2 - rh // 2
    x2, y2 = x1 + rw, y1 + rh
    color = (185, 28, 28) if hazard else (26, 127, 55)  # Red or Green (BGR)
    thickness = 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Label
    label = "Machine"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(label, font, 1, 2)[0]
    label_x = x1 + (rw - label_size[0]) // 2
    label_y = y1 - 10 if y1 - 10 > 20 else y1 + 30
    cv2.putText(frame, label, (label_x, label_y), font, 1, color, 2, cv2.LINE_AA)
    return frame

if input_source == "Webcam":
    run = st.checkbox("Start Webcam", value=False)
    FRAME_WINDOW = st.image([])
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break
            hazard, mask, red_pixels = detect_hazard(frame, hazard_threshold)
            status = "STOPPED (Hazard Detected)" if hazard else "Safe"
            color = (0, 0, 255) if hazard else (0, 200, 0)
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Red Pixels: {red_pixels}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            # --- Highlight machine ---
            frame = draw_machine_highlight(frame, hazard)
            # --- Live status circle ---
            circle_emoji = "🔴" if hazard else "🟢"
            status_circle_placeholder.markdown(f"""
                <div style='display:flex;justify-content:center;align-items:center;margin-bottom:10px;'>
                    <span style='font-size:3.5em;'>{circle_emoji}</span>
                </div>
            """, unsafe_allow_html=True)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
            if hazard:
                st.session_state['hazard_count'] += 1
                status_placeholder.markdown(f"<div style='background:#ffeaea;color:#b91c1c;padding:12px;border-radius:8px;font-size:1.2em;text-align:center;'>🚨 <b>Machine Status: STOPPED (Hazard Detected)</b></div>", unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f"<div style='background:#e6f9ed;color:#1a7f37;padding:12px;border-radius:8px;font-size:1.2em;text-align:center;'>✅ <b>Machine Status: Safe</b></div>", unsafe_allow_html=True)
            hazard_count_placeholder.markdown(f"<div style='text-align:center;margin-top:10px;'><span style='background:#b91c1c;color:white;padding:6px 18px;border-radius:20px;font-size:1.1em;'>Hazard Events: {st.session_state['hazard_count']}</span></div>", unsafe_allow_html=True)
            # Snapshot button
            if snapshot_placeholder.button("📸 Snapshot", key=int(time.time()*1000)):
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                img.save(buf.name)
                st.session_state['last_snapshot'] = buf.name
                st.success("Snapshot saved! Scroll down to download.")
            # Show download if snapshot exists
            if st.session_state['last_snapshot']:
                with open(st.session_state['last_snapshot'], "rb") as file:
                    st.download_button("Download Last Snapshot", file, file_name="snapshot.jpg", mime="image/jpeg")
            time.sleep(0.05)
        cap.release()
else:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hazard, mask, red_pixels = detect_hazard(frame, hazard_threshold)
        status = "STOPPED (Hazard Detected)" if hazard else "Safe"
        color = (0, 0, 255) if hazard else (0, 200, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Red Pixels: {red_pixels}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # --- Highlight machine ---
        frame = draw_machine_highlight(frame, hazard)
        # --- Live status circle ---
        circle_emoji = "🔴" if hazard else "🟢"
        status_circle_placeholder.markdown(f"""
            <div style='display:flex;justify-content:center;align-items:center;margin-bottom:10px;'>
                <span style='font-size:3.5em;'>{circle_emoji}</span>
            </div>
        """, unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', caption="Processed Image")
        if hazard:
            st.session_state['hazard_count'] += 1
            status_placeholder.markdown(f"<div style='background:#ffeaea;color:#b91c1c;padding:12px;border-radius:8px;font-size:1.2em;text-align:center;'>🚨 <b>Machine Status: STOPPED (Hazard Detected)</b></div>", unsafe_allow_html=True)
        else:
            status_placeholder.markdown(f"<div style='background:#e6f9ed;color:#1a7f37;padding:12px;border-radius:8px;font-size:1.2em;text-align:center;'>✅ <b>Machine Status: Safe</b></div>", unsafe_allow_html=True)
        hazard_count_placeholder.markdown(f"<div style='text-align:center;margin-top:10px;'><span style='background:#b91c1c;color:white;padding:6px 18px;border-radius:20px;font-size:1.1em;'>Hazard Events: {st.session_state['hazard_count']}</span></div>", unsafe_allow_html=True)
        # Snapshot button
        if snapshot_placeholder.button("📸 Snapshot", key=int(time.time()*1000)):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img.save(buf.name)
            st.session_state['last_snapshot'] = buf.name
            st.success("Snapshot saved! Scroll down to download.")
        # Show download if snapshot exists
        if st.session_state['last_snapshot']:
            with open(st.session_state['last_snapshot'], "rb") as file:
                st.download_button("Download Last Snapshot", file, file_name="snapshot.jpg", mime="image/jpeg") 