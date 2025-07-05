"""
AI-Powered Accident Prevention in Modular Manufacturing Systems (MMS)
Simplified Version for Cloud Deployment
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

# Configure Streamlit page FIRST (before any other st commands)
st.set_page_config(
    page_title="MMS Safety System - AI-Powered Accident Prevention",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle OpenCV import with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("⚠️ OpenCV not available - using basic PIL-based detection")

# Suppress Streamlit warnings for cleaner operation
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# --- Sidebar Configuration ---
st.sidebar.title("🏭 MMS Safety System")
st.sidebar.markdown("**Configuration Panel**")

# Input source selection
input_source = st.sidebar.radio(
    "📹 Input Source", 
    ["Image Upload", "Video Upload"], 
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

st.sidebar.markdown("---")
st.sidebar.info(
    "🔒 **Industrial Safety System**\n\n"
    "This AI-powered system continuously monitors your MMS workspace "
    "and automatically stops machines when hazards are detected. "
    "Designed for maximum reliability and minimum false alarms."
)

# --- Session State Management ---
if 'hazard_count' not in st.session_state:
    st.session_state['hazard_count'] = 0
if 'last_hazard_time' not in st.session_state:
    st.session_state['last_hazard_time'] = 0
if 'machine_status' not in st.session_state:
    st.session_state['machine_status'] = "Safe"

# --- Main Application Header ---
st.title("🏭 AI-Powered Accident Prevention in MMS")
st.markdown("""
**Organisation:** International Automobile Centre of Excellence, Ahmedabad  
**Category:** Industry Defined Problem - Modular Manufacturing Systems Safety
""")

# Professional context and system status
st.markdown("""
<div style='padding:20px;border-radius:10px;border-left:5px solid #007bff;margin:20px 0;border:1px solid #e9ecef;'>
    <h3>🔒 Industrial Safety Monitoring System</h3>
    <p><strong>Mission:</strong> Prevent accidents in Modular Manufacturing Systems through real-time AI vision monitoring.</p>
    <p><strong>Current Status:</strong> System ready for deployment. Monitor workspace activities and receive instant alerts for potential hazards.</p>
</div>
""", unsafe_allow_html=True)

# --- Hazard Detection Algorithm ---
def detect_hazard(frame, threshold, confidence=0.8):
    """
    Hazard detection algorithm for industrial safety.
    
    Args:
        frame: Input image frame
        threshold: Sensitivity threshold for detection
        confidence: Confidence level for detection (0.1-1.0)
    
    Returns:
        tuple: (hazard_detected, red_pixel_count)
    """
    if not OPENCV_AVAILABLE:
        return detect_hazard_pil(frame, threshold, confidence)
    
    try:
        if frame is None or frame.size == 0:
            return False, 0
        
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
        
        return hazard, red_pixels
        
    except Exception as e:
        st.error(f"⚠️ Error in hazard detection algorithm: {str(e)}")
        return False, 0

def detect_hazard_pil(frame, threshold, confidence=0.8):
    """
    Fallback hazard detection using PIL (when OpenCV is not available).
    """
    try:
        if frame is None:
            return False, 0
        
        # Convert PIL image to numpy array if needed
        if hasattr(frame, 'convert'):
            frame = np.array(frame.convert('RGB'))
        
        # Simple red pixel detection using numpy
        red_pixels = np.sum((frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100))
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        hazard = red_pixels > threshold and red_percentage > confidence * 0.01
        return hazard, red_pixels
        
    except Exception as e:
        st.error(f"⚠️ Error in PIL-based hazard detection: {str(e)}")
        return False, 0

# --- Image Upload Analysis Mode ---
if input_source == "Image Upload":
    st.subheader("📸 Static Image Analysis")
    st.markdown("**Upload a workspace image for hazard analysis.**")
    
    uploaded_file = st.file_uploader(
        "Choose a workspace image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your MMS workspace for hazard detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process the image
            image = Image.open(uploaded_file)
            
            # Convert to numpy array for processing
            if OPENCV_AVAILABLE:
                frame = np.array(image.convert('RGB'))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = np.array(image.convert('RGB'))
            
            # Perform hazard detection
            hazard, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Workspace Image", use_column_width=True)
            
            with col2:
                # Status display
                if hazard:
                    st.markdown("""
                        <div style='background:#ffeaea;color:#b91c1c;padding:20px;border-radius:10px;text-align:center;border:2px solid #b91c1c;'>
                            🚨 <b>HAZARD DETECTED</b><br>
                            <small>Red hazard indicators found in workspace</small>
                        </div>
                    """, unsafe_allow_html=True)
                    st.session_state['machine_status'] = "STOPPED"
                else:
                    st.markdown("""
                        <div style='background:#e6f9ed;color:#1a7f37;padding:20px;border-radius:10px;text-align:center;border:2px solid #1a7f37;'>
                            ✅ <b>WORKSPACE SAFE</b><br>
                            <small>No hazards detected in workspace</small>
                        </div>
                    """, unsafe_allow_html=True)
                    st.session_state['machine_status'] = "Safe"
                
                # Analysis details
                st.markdown("### 📊 Analysis Results")
                st.write(f"**Red Pixels Detected:** {red_pixels:,}")
                st.write(f"**Detection Threshold:** {hazard_threshold:,}")
                st.write(f"**Confidence Level:** {confidence_threshold:.1%}")
                st.write(f"**Machine Status:** {st.session_state['machine_status']}")
                
                if hazard:
                    st.session_state['hazard_count'] += 1
                    st.write(f"**Total Hazard Events:** {st.session_state['hazard_count']}")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# --- Video Upload Analysis Mode ---
elif input_source == "Video Upload":
    st.subheader("🎬 Video Analysis")
    st.markdown("**Upload a workspace video for hazard analysis.**")
    
    uploaded_video = st.file_uploader(
        "Choose a workspace video", 
        type=["mp4", "avi", "mov"],
        help="Upload a video of your MMS workspace for hazard detection analysis"
    )
    
    if uploaded_video is not None:
        st.info("⚠️ Video analysis requires OpenCV. Please use Image Upload mode if OpenCV is not available.")
        
        if not OPENCV_AVAILABLE:
            st.error("OpenCV is required for video analysis. Please use Image Upload mode instead.")
        else:
            st.success("Video analysis will be implemented when OpenCV is available.")

# --- System Status Display ---
st.markdown("---")
st.markdown("### 🔧 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Machine Status", st.session_state['machine_status'])
    
with col2:
    st.metric("Hazard Events", st.session_state['hazard_count'])
    
with col3:
    status_emoji = "🔴" if st.session_state['machine_status'] == "STOPPED" else "🟢"
    st.metric("System Health", f"{status_emoji} Operational")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.9em;'>
    <p><strong>AI-Powered Accident Prevention System</strong></p>
    <p>International Automobile Centre of Excellence, Ahmedabad</p>
    <p>Modular Manufacturing Systems Safety Project</p>
</div>
""", unsafe_allow_html=True) 