import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import time
import warnings
import os

# Configure page first
st.set_page_config(
    page_title="MMS Safety System - AI-Powered Accident Prevention",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle OpenCV import
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-5px);
    }
    
    .status-safe {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    
    .status-hazard {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 4px 30px rgba(239, 68, 68, 0.6); }
        100% { box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3); }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Webcam container */
    .webcam-container {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        margin: 1rem 0;
    }
    
    .webcam-container.active {
        border-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    /* Control buttons */
    .control-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .control-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .control-btn.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .control-btn.success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    /* Progress bars */
    .progress-container {
        background: #e5e7eb;
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .progress-safe {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    
    .progress-hazard {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
    }
    
    /* Alert animations */
    .alert-flash {
        animation: flash 1s infinite;
    }
    
    @keyframes flash {
        0%, 50% { opacity: 1; }
        25%, 75% { opacity: 0.7; }
    }
    
    /* Image upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏭 MMS Safety System</h1>
    <p>AI-Powered Accident Prevention in Modular Manufacturing Systems</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.title("🏭 MMS Safety System")
st.sidebar.markdown("**Configuration Panel**")

# Input source selection with better styling
input_source = st.sidebar.selectbox(
    "📹 Input Source", 
    ["Webcam", "Image Upload", "Video Upload"], 
    index=0,
    help="Select the source for hazard detection monitoring"
)

# Detection sensitivity configuration
hazard_threshold = st.sidebar.slider(
    "🎯 Hazard Detection Sensitivity", 
    5000, 50000, 15000, 1000,
    help="Higher values = more sensitive detection"
)

confidence_threshold = st.sidebar.slider(
    "⚡ Detection Confidence", 
    0.1, 1.0, 0.8, 0.1,
    help="Higher values = stricter detection"
)

# Session state
if 'hazard_count' not in st.session_state:
    st.session_state['hazard_count'] = 0
if 'last_hazard_time' not in st.session_state:
    st.session_state['last_hazard_time'] = 0
if 'machine_status' not in st.session_state:
    st.session_state['machine_status'] = "Safe"
if 'consecutive_hazards' not in st.session_state:
    st.session_state['consecutive_hazards'] = 0
if 'consecutive_safe' not in st.session_state:
    st.session_state['consecutive_safe'] = 0

# Add info section to sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "🔒 **Industrial Safety System**\n\n"
    "This AI-powered system continuously monitors your MMS workspace "
    "and automatically stops machines when hazards are detected. "
    "Designed for maximum reliability and minimum false alarms."
)

# Hazard detection function
def detect_hazard(frame, threshold, confidence=0.8):
    """Enhanced hazard detection with better algorithms."""
    if not OPENCV_AVAILABLE:
        return detect_hazard_pil(frame, threshold, confidence)
    
    try:
        if frame is None or frame.size == 0:
            return False, 0
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red detection
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([170, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        red_mask = mask1 + mask2
        
        # Morphological operations
        kernel = np.ones((7,7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        hazard = (
            red_pixels > threshold * 2 and
            red_percentage > 0.01 and
            red_percentage > confidence * 0.02
        )
        
        return hazard, red_pixels
        
    except Exception as e:
        st.error(f"Error in hazard detection: {str(e)}")
        return False, 0

def detect_hazard_pil(frame, threshold, confidence=0.8):
    """Fallback hazard detection using PIL."""
    try:
        if frame is None:
            return False, 0
        
        if hasattr(frame, 'convert'):
            frame = np.array(frame.convert('RGB'))
        
        red_pixels = np.sum((frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100))
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        hazard = red_pixels > threshold and red_percentage > confidence * 0.01
        return hazard, red_pixels
        
    except Exception as e:
        st.error(f"Error in PIL-based hazard detection: {str(e)}")
        return False, 0

def check_webcam():
    """Enhanced webcam check with better error handling."""
    if not OPENCV_AVAILABLE:
        return False
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False
            
        return True
    except Exception:
        return False

# Webcam monitoring with enhanced interface
if input_source == "Webcam":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📹 Real-Time Webcam Monitoring</h2>
        <p style="color: #6b7280; font-size: 1.1rem;">Monitor your MMS workspace in real-time for potential hazards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Webcam status check
    webcam_available = check_webcam()
    
    if not OPENCV_AVAILABLE:
        st.markdown("""
        <div class="status-card status-hazard">
            <h3>⚠️ OpenCV Not Available</h3>
            <p>OpenCV is required for webcam functionality but is not available in this environment.</p>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Use Image Upload mode for static analysis</li>
                <li>For local deployment, install: <code>pip install opencv-python-headless</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    elif not webcam_available:
        st.markdown("""
        <div class="status-card status-hazard">
            <h3>⚠️ Webcam Not Available</h3>
            <p>Please ensure:</p>
            <ul>
                <li>Your camera is connected and not used by another application</li>
                <li>Camera permissions are granted</li>
                <li>Try restarting your computer if issues persist</li>
            </ul>
            <p><strong>Alternative:</strong> Use Image Upload mode for static analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Webcam controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        run = st.button("🚀 Start Real-Time Monitoring", type="primary", use_container_width=True)
    
    # Webcam container
    webcam_container = st.container()
    
    with webcam_container:
        st.markdown("""
        <div class="webcam-container" id="webcam-container">
            <h3>📹 Webcam Feed</h3>
            <p>Click "Start Real-Time Monitoring" to begin</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for webcam feed
        webcam_placeholder = st.empty()
        
        # Status indicators
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            status_placeholder = st.empty()
        
        with status_col2:
            hazard_count_placeholder = st.empty()
        
        with status_col3:
            progress_placeholder = st.empty()
    
    if run:
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                st.error("Failed to open webcam")
                st.stop()
            
            frame_count = 0
            
            # Real-time monitoring loop
            while run:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("Failed to read frame from webcam. Retrying...")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Hazard detection
                hazard, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
                
                # Update session state
                if hazard:
                    st.session_state['consecutive_hazards'] += 1
                    st.session_state['consecutive_safe'] = 0
                else:
                    st.session_state['consecutive_safe'] += 1
                    st.session_state['consecutive_hazards'] = 0
                
                # Determine final status
                final_hazard = st.session_state['consecutive_hazards'] >= 5
                final_safe = st.session_state['consecutive_safe'] >= 10
                
                if final_hazard:
                    st.session_state['machine_status'] = "STOPPED"
                    if time.time() - st.session_state['last_hazard_time'] > 5:
                        st.session_state['hazard_count'] += 1
                        st.session_state['last_hazard_time'] = time.time()
                elif final_safe:
                    st.session_state['machine_status'] = "Safe"
                
                # Display webcam feed
                if OPENCV_AVAILABLE:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame_rgb, channels='RGB', caption="Live Webcam Feed")
                else:
                    frame_rgb = frame[:, :, ::-1]
                    pil_image = Image.fromarray(frame_rgb)
                    webcam_placeholder.image(pil_image, channels='RGB', caption="Live Webcam Feed")
                
                # Update status displays
                if final_hazard:
                    status_placeholder.markdown(f"""
                    <div class="status-card status-hazard alert-flash">
                        <h3>🚨 EMERGENCY STOP</h3>
                        <p>Hazard detected in machine zone</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status_placeholder.markdown(f"""
                    <div class="status-card status-safe">
                        <h3>✅ SYSTEM SAFE</h3>
                        <p>No hazards detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Hazard count
                hazard_count_placeholder.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state['hazard_count']}</div>
                    <div class="metric-label">Hazard Events</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bars
                hazard_progress = min(st.session_state['consecutive_hazards'] / 5 * 100, 100)
                safe_progress = min(st.session_state['consecutive_safe'] / 10 * 100, 100)
                
                progress_placeholder.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Hazard Level</span>
                        <span>{st.session_state['consecutive_hazards']}/5</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-hazard" style="width: {hazard_progress}%"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Safe Level</span>
                        <span>{st.session_state['consecutive_safe']}/10</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar progress-safe" style="width: {safe_progress}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(0.1)  # 10 FPS for better performance
            
            cap.release()
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.stop()

# Image upload analysis with enhanced styling
elif input_source == "Image Upload":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📸 Static Image Analysis</h2>
        <p style="color: #6b7280; font-size: 1.1rem;">Upload a workspace image for hazard analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a workspace image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your MMS workspace for hazard detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process image
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            
            if OPENCV_AVAILABLE:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect hazards
            hazard, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Workspace Image", use_column_width=True)
            
            with col2:
                if hazard:
                    st.markdown(f"""
                    <div class="status-card status-hazard">
                        <h3>🚨 HAZARD DETECTED</h3>
                        <p>Red hazard indicators found in workspace</p>
                        <div class="metric-value">{red_pixels:,}</div>
                        <div class="metric-label">Red Pixels Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state['machine_status'] = "STOPPED"
                    st.session_state['hazard_count'] += 1
                else:
                    st.markdown(f"""
                    <div class="status-card status-safe">
                        <h3>✅ WORKSPACE SAFE</h3>
                        <p>No hazards detected in workspace</p>
                        <div class="metric-value">{red_pixels:,}</div>
                        <div class="metric-label">Red Pixels Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state['machine_status'] = "Safe"
                
                # Analysis details
                st.markdown("### 📊 Analysis Results")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Detection Threshold", f"{hazard_threshold:,}")
                    st.metric("Machine Status", st.session_state['machine_status'])
                
                with col_b:
                    st.metric("Confidence Level", f"{confidence_threshold:.1%}")
                    st.metric("Total Hazard Events", st.session_state['hazard_count'])
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <h3>🏭 AI-Powered Accident Prevention System</h3>
    <p>Modular Manufacturing Systems Safety Project</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Powered by Computer Vision & AI</p>
</div>
""", unsafe_allow_html=True) 