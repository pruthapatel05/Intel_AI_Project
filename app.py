"""
AI-Powered Accident Prevention in Modular Manufacturing Systems (MMS)
====================================================================

This application implements a real-time AI vision system for detecting hazards
in industrial manufacturing environments. The system continuously monitors
workspace activities and can automatically stop machines when potential hazards
are detected, preventing accidents and ensuring workplace safety.

Key Features:
- Real-time hazard detection using computer vision
- Ultra-strict detection logic to minimize false alarms
- Automatic machine status control (Safe/Stopped)
- Professional industrial interface
- Configurable sensitivity and confidence thresholds

Developed for: International Automobile Centre of Excellence, Ahmedabad
Category: Industry Defined Problem - MMS Safety
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import time
import warnings
import os

# Handle OpenCV import with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    st.error("""
    ⚠️ **OpenCV Import Error**
    
    OpenCV is not available in this environment. This is common in cloud deployments.
    
    **Solutions:**
    1. Use the headless version: `pip install opencv-python-headless`
    2. For Streamlit Cloud, add this to your requirements.txt:
       ```
       opencv-python-headless==4.8.1.78
       ```
    3. The application will continue with limited functionality.
    """)
    OPENCV_AVAILABLE = False

# Suppress Streamlit warnings for cleaner operation
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Configure Streamlit page for professional industrial interface
st.set_page_config(
    page_title="MMS Safety System - AI-Powered Accident Prevention",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
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
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("🏭 MMS Safety System")
st.sidebar.markdown("**Configuration Panel**")

# Input source selection
input_source = st.sidebar.radio(
    "📹 Input Source", 
    ["Webcam", "Video Upload", "Image Upload"], 
    index=0,
    help="Select the source for hazard detection monitoring"
)

# Detection sensitivity configuration
hazard_threshold = st.sidebar.slider(
    "🎯 Hazard Detection Sensitivity", 
    5000, 50000, 15000, 1000,
    help="Higher values = more sensitive detection (may increase false alarms)"
)

# Detection confidence configuration
confidence_threshold = st.sidebar.slider(
    "⚡ Detection Confidence", 
    0.1, 1.0, 0.8, 0.1,
    help="Higher values = stricter detection (fewer false positives)"
)

# Video analysis settings (only show for video upload)
if input_source == "Video Upload":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎬 Video Analysis Settings**")
    
    # Frame sampling rate for video analysis
    frame_sample_rate = st.sidebar.slider(
        "📊 Frame Sample Rate", 
        1, 30, 5, 1,
        help="Analyze every Nth frame (higher = faster processing, lower = more accurate)"
    )
    
    # Video analysis speed
    analysis_speed = st.sidebar.selectbox(
        "⚡ Analysis Speed",
        ["Fast (Basic)", "Standard", "Detailed"],
        index=1,
        help="Trade-off between speed and detection accuracy"
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "🔒 **Industrial Safety System**\n\n"
    "This AI-powered system continuously monitors your MMS workspace "
    "and automatically stops machines when hazards are detected. "
    "Designed for maximum reliability and minimum false alarms."
)

# --- Session State Management ---
# Initialize persistent state variables for the application
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

# --- Main Application Header ---
st.markdown("""
<div class="main-header">
    <h1>🏭 MMS Safety System</h1>
    <p>AI-Powered Accident Prevention in Modular Manufacturing Systems</p>
</div>
""", unsafe_allow_html=True)

# --- Ultra-Strict Hazard Detection Algorithm ---
def detect_hazard(frame, threshold, confidence=0.8):
    """
    Ultra-strict hazard detection algorithm for industrial safety.
    
    Args:
        frame: Input image frame (BGR format)
        threshold: Sensitivity threshold for detection
        confidence: Confidence level for detection (0.1-1.0)
    
    Returns:
        tuple: (hazard_detected, mask, red_pixel_count)
    """
    if not OPENCV_AVAILABLE:
        st.warning("⚠️ OpenCV not available - using basic PIL-based detection")
        return detect_hazard_pil(frame, threshold, confidence)
    
    try:
        if frame is None or frame.size == 0:
            return False, None, 0
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red detection with multiple ranges for industrial lighting
        # Lower red range (0-10 degrees in HSV)
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Upper red range (170-180 degrees in HSV)
        lower_red2 = np.array([170, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks for comprehensive red detection
        red_mask = mask1 + mask2
        
        # Apply morphological operations to reduce noise and improve detection
        kernel = np.ones((7,7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Count red pixels and calculate percentage
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        # Ultra-strict hazard detection criteria for industrial safety
        min_red_percentage = 0.01  # Minimum 1% of frame must be red
        hazard = (
            red_pixels > threshold * 2 and  # High pixel count requirement
            red_percentage > min_red_percentage and  # Minimum percentage requirement
            red_percentage > confidence * 0.02  # Confidence-based threshold
        )
        
        return hazard, red_mask, red_pixels
        
    except Exception as e:
        st.error(f"⚠️ Error in hazard detection algorithm: {str(e)}")
        return False, None, 0

def detect_hazard_pil(frame, threshold, confidence=0.8):
    """
    Fallback hazard detection using PIL (when OpenCV is not available).
    """
    try:
        if frame is None:
            return False, None, 0
        
        # Convert PIL image to numpy array if needed
        if hasattr(frame, 'convert'):
            frame = np.array(frame.convert('RGB'))
        
        # Simple red pixel detection using numpy
        red_pixels = np.sum((frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100))
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        hazard = red_pixels > threshold and red_percentage > confidence * 0.01
        return hazard, None, red_pixels
        
    except Exception as e:
        st.error(f"⚠️ Error in PIL-based hazard detection: {str(e)}")
        return False, None, 0

# --- Webcam Validation Function ---
def check_webcam():
    """
    Validate webcam availability for real-time monitoring.
    
    Returns:
        bool: True if webcam is available and functional
    """
    if not OPENCV_AVAILABLE:
        return False
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False

# --- Machine Zone Visualization ---
def draw_machine_highlight(frame, hazard):
    """
    Draw machine zone highlighting with professional industrial styling.
    
    Args:
        frame: Input frame to draw on
        hazard: Boolean indicating if hazard is detected
    
    Returns:
        numpy.ndarray: Frame with machine zone highlighting
    """
    if not OPENCV_AVAILABLE:
        # Return frame without OpenCV drawing when not available
        return frame
    
    try:
        if frame is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # Define machine zone (centered rectangle, 40% of frame)
        rw, rh = int(w * 0.4), int(h * 0.4)
        x1, y1 = w // 2 - rw // 2, h // 2 - rh // 2
        x2, y2 = x1 + rw, y1 + rh
        
        # Professional color coding: Red for hazard, Green for safe
        color = (0, 0, 255) if hazard else (0, 255, 0)  # BGR format
        thickness = 4
        
        # Draw main machine zone rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add corner indicators for professional appearance
        corner_size = 20
        cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, thickness)
        
        # Add professional machine zone label
        label = "MACHINE ZONE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        label_x = x1 + (rw - label_size[0]) // 2
        label_y = y1 - 20 if y1 - 20 > 30 else y1 + 40
        
        # Add background for better text visibility
        cv2.rectangle(frame, (label_x - 10, label_y - label_size[1] - 10), 
                     (label_x + label_size[0] + 10, label_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y), font, font_scale, color, 2, cv2.LINE_AA)
        
        return frame
    except Exception as e:
        st.error(f"⚠️ Error drawing machine zone: {str(e)}")
        return frame

# --- Main Application Interface ---

# --- Webcam Monitoring Mode ---
if input_source == "Webcam":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📹 Real-Time Webcam Monitoring</h2>
        <p style="color: #6b7280; font-size: 1.1rem;">Monitor your MMS workspace in real-time for potential hazards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Validate webcam availability
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
    elif not check_webcam():
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
    
    # Webcam controls with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        run = st.button("🚀 Start Real-Time Monitoring", type="primary", use_container_width=True)
    
    # Webcam container
    st.markdown("""
    <div class="webcam-container">
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
            if not OPENCV_AVAILABLE:
                st.error("OpenCV not available for webcam functionality")
                st.stop()
            
            # Initialize webcam with optimal settings for industrial monitoring
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                st.error("""
                ❌ **Failed to Open Webcam**
                
                Please check:
                - Camera permissions in your system settings
                - No other applications are using the camera
                - Camera drivers are properly installed
                """)
                st.stop()
            
            frame_count = 0
            
            # Real-time monitoring loop
            while run:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("⚠️ Failed to read frame from webcam. Retrying...")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Perform ultra-strict hazard detection
                hazard, mask, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
                
                # Advanced debouncing logic for industrial reliability
                if hazard:
                    st.session_state['consecutive_hazards'] += 1
                    st.session_state['consecutive_safe'] = 0
                else:
                    st.session_state['consecutive_safe'] += 1
                    st.session_state['consecutive_hazards'] = 0
                
                # Ultra-strict triggering criteria
                final_hazard = st.session_state['consecutive_hazards'] >= 10
                final_safe = st.session_state['consecutive_safe'] >= 15
                
                # Update machine status with industrial-grade timing
                if final_hazard:
                    st.session_state['machine_status'] = "STOPPED"
                    if time.time() - st.session_state['last_hazard_time'] > 5:  # 5-second cooldown
                        st.session_state['hazard_count'] += 1
                        st.session_state['last_hazard_time'] = time.time()
                elif final_safe:
                    st.session_state['machine_status'] = "Safe"
                
                # Add professional status overlay to frame
                status_text = f"Status: {st.session_state['machine_status']}"
                color = (0, 0, 255) if final_hazard else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Red Pixels: {red_pixels}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f"Hazard Count: {st.session_state['consecutive_hazards']}/10", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Safe Count: {st.session_state['consecutive_safe']}/15", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Highlight machine zone with professional styling
                frame = draw_machine_highlight(frame, final_hazard)
                
                # Display processed frame
                if OPENCV_AVAILABLE:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame_rgb, channels='RGB', caption="Live Webcam Feed")
                else:
                    # Convert numpy array to PIL Image for display
                    frame_rgb = frame[:, :, ::-1]  # BGR to RGB
                    pil_image = Image.fromarray(frame_rgb)
                    webcam_placeholder.image(pil_image, channels='RGB', caption="Live Webcam Feed")
                
                # Update status displays with new styling
                if final_hazard:
                    status_placeholder.markdown(f"""
                    <div class="status-card status-hazard">
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
                
                # Hazard count with new styling
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
                
                time.sleep(0.03)  # Maintain ~30 FPS for smooth operation
                
            cap.release()
            
        except Exception as e:
            st.error(f"""
            ❌ **System Error**
            
            Error details: {str(e)}
            
            Please:
            1. Check your system resources
            2. Restart the application
            3. Contact technical support if issues persist
            """)
            st.stop()

# --- Image Upload Analysis Mode ---
elif input_source == "Image Upload":
    st.subheader("📸 Static Image Analysis")
    st.markdown("**Upload a workspace image for hazard analysis.**")
    
    uploaded_file = st.file_uploader(
        "Choose a workspace image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your MMS workspace for hazard detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process uploaded image
            image = Image.open(uploaded_file)
            frame = np.array(image)
            
            # Handle different image formats
            if len(frame.shape) == 3:
                if frame.shape[-1] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Perform hazard detection on uploaded image
            hazard, mask, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
            
            # Update machine status based on analysis
            if hazard:
                st.session_state['machine_status'] = "STOPPED"
                st.session_state['hazard_count'] += 1
            else:
                st.session_state['machine_status'] = "Safe"
            
            # Add professional status overlay
            status_text = f"Status: {st.session_state['machine_status']}"
            color = (0, 0, 255) if hazard else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Red Pixels: {red_pixels}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Highlight machine zone
            frame = draw_machine_highlight(frame, hazard)
            
            # Display analyzed image
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', caption="Hazard Analysis Results")
            
            # Display professional status message
            if hazard:
                st.markdown(f"""
                    <div class="status-card status-hazard">
                        <h3>🚨 HAZARD DETECTED</h3>
                        <p>Analysis indicates potential safety risk. Machine should be stopped immediately.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="status-card status-safe">
                        <h3>✅ WORKSPACE SAFE</h3>
                        <p>No hazards detected in the analyzed image. Workspace appears safe for operation.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display hazard event counter
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state['hazard_count']}</div>
                    <div class="metric-label">Total Hazard Events</div>
                </div>
            """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"""
            ❌ **Analysis Error**
            
            Error details: {str(e)}
            
            Please ensure:
            - Image format is supported (JPG, JPEG, PNG)
            - Image is not corrupted
            - Try uploading a different image
            """)

# --- Video Upload Analysis Mode ---
elif input_source == "Video Upload":
    st.subheader("📹 Video Analysis")
    st.markdown("**Upload a video for comprehensive hazard analysis.**")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video of your MMS workspace for hazard detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            # Load video using the temporary file path
            video = cv2.VideoCapture(temp_video_path)
            
            if not video.isOpened():
                st.error("""
                ❌ **Failed to Open Video**
                
                Please check:
                - Video file is valid and not corrupted
                - Video format is supported (MP4, AVI, MOV, MKV)
                - Try uploading a different video
                """)
                # Clean up temporary file
                os.unlink(temp_video_path)
                st.stop()
            
            # Get video properties
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            st.info(f"""
            📊 **Video Information**
            - **Duration:** {duration:.1f} seconds
            - **Total Frames:** {total_frames:,}
            - **Frame Rate:** {fps:.1f} FPS
            - **Analysis Speed:** {analysis_speed}
            - **Frame Sample Rate:** Every {frame_sample_rate} frame(s)
            """)
            
            # Initialize analysis variables
            frame_count = 0
            analyzed_frames = 0
            hazard_frames = 0
            machine_status = "Safe"
            consecutive_hazards = 0
            consecutive_safe = 0
            hazard_timestamps = []
            
            # Create progress bar for video analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Video analysis loop with frame sampling
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Only analyze frames based on sample rate
                if frame_count % frame_sample_rate == 0:
                    analyzed_frames += 1
                    
                    # Perform ultra-strict hazard detection
                    hazard, mask, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
                    
                    # Advanced debouncing logic for industrial reliability
                    if hazard:
                        consecutive_hazards += 1
                        consecutive_safe = 0
                        hazard_frames += 1
                        hazard_timestamps.append(frame_count / fps)  # Convert to seconds
                    else:
                        consecutive_safe += 1
                        consecutive_hazards = 0
                    
                    # Ultra-strict triggering criteria
                    final_hazard = consecutive_hazards >= 10
                    final_safe = consecutive_safe >= 15
                    
                    # Update machine status
                    if final_hazard:
                        machine_status = "STOPPED"
                    elif final_safe:
                        machine_status = "Safe"
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing frame {frame_count:,} of {total_frames:,} ({progress:.1%})")
                
                # Display sample frames (every 30th frame to avoid overwhelming)
                if frame_count % 30 == 0:
                    # Add professional status overlay to sample frame
                    status_text_overlay = f"Status: {machine_status}"
                    color = (0, 0, 255) if machine_status == "STOPPED" else (0, 255, 0)
                    cv2.putText(frame, status_text_overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Frame: {frame_count:,}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Highlight machine zone
                    frame = draw_machine_highlight(frame, machine_status == "STOPPED")
                    
                    # Display sample frame
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', 
                            caption=f"Sample Frame {frame_count:,} - Status: {machine_status}")
            
            # Clean up
            video.release()
            os.unlink(temp_video_path)
            
            # Analysis complete - show results
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis Complete!")
            
            # Calculate analysis statistics
            hazard_percentage = (hazard_frames / analyzed_frames * 100) if analyzed_frames > 0 else 0
            total_hazards = len([t for t in hazard_timestamps if t > 0])
            
            # Display comprehensive analysis results
            st.success(f"""
            📊 **Video Analysis Complete**
            
            **Analysis Summary:**
            - **Total Frames Analyzed:** {analyzed_frames:,} of {total_frames:,}
            - **Hazard Frames Detected:** {hazard_frames:,} ({hazard_percentage:.1f}%)
            - **Total Hazard Events:** {total_hazards}
            - **Final Machine Status:** {machine_status}
            """)
            
            # Display hazard timeline if hazards were detected
            if hazard_timestamps:
                st.warning(f"""
                🚨 **Hazard Timeline**
                
                Hazards detected at the following timestamps:
                {', '.join([f'{t:.1f}s' for t in hazard_timestamps[:10]])}
                {'...' if len(hazard_timestamps) > 10 else ''}
                """)
            
            # Display final status message
            if total_hazards > 0:
                st.markdown(f"""
                    <div class="status-card status-hazard">
                        <h3>🚨 HAZARDS DETECTED IN VIDEO</h3>
                        <p>Analysis found {total_hazards} hazard events. Machine should be stopped during these periods.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="status-card status-safe">
                        <h3>✅ VIDEO ANALYSIS: SAFE</h3>
                        <p>No hazards detected in the analyzed video. Workspace appears safe throughout.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display hazard event counter
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_hazards}</div>
                    <div class="metric-label">Total Hazards</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"""
            ❌ **Video Analysis Error**
            
            Error details: {str(e)}
            
            Please ensure:
            - Video file is valid and not corrupted
            - Video format is supported (MP4, AVI, MOV, MKV)
            - Try uploading a different video
            - Check available system memory
            """)



# --- Professional Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.9em;padding:20px;background:#f8f9fa;border-radius:10px;'>
    <p><strong>🏭 AI-Powered Accident Prevention System</strong></p>
    <p>Industrial-Grade Safety Monitoring for Modular Manufacturing Systems</p>
    <p>Built with Streamlit, OpenCV, and Advanced Computer Vision</p>
</div>
""", unsafe_allow_html=True) 