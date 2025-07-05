import streamlit as st
import numpy as np
from PIL import Image

# Configure page first
st.set_page_config(
    page_title="MMS Safety System",
    page_icon="🏭",
    layout="wide"
)

# Handle OpenCV import
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Main app
st.title("🏭 AI-Powered Accident Prevention in MMS")
st.markdown("**Organisation:** International Automobile Centre of Excellence, Ahmedabad")

# Sidebar
st.sidebar.title("🏭 MMS Safety System")
hazard_threshold = st.sidebar.slider("Hazard Detection Sensitivity", 5000, 50000, 15000, 1000)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.8, 0.1)

# Session state
if 'hazard_count' not in st.session_state:
    st.session_state['hazard_count'] = 0
if 'machine_status' not in st.session_state:
    st.session_state['machine_status'] = "Safe"

# Hazard detection function
def detect_hazard(frame, threshold, confidence=0.8):
    """Simple hazard detection using red pixel counting."""
    try:
        if frame is None:
            return False, 0
        
        # Convert to numpy array if needed
        if hasattr(frame, 'convert'):
            frame = np.array(frame.convert('RGB'))
        
        # Simple red pixel detection
        red_pixels = np.sum((frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100))
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        hazard = red_pixels > threshold and red_percentage > confidence * 0.01
        return hazard, red_pixels
        
    except Exception as e:
        st.error(f"Error in hazard detection: {str(e)}")
        return False, 0

# Main interface
st.subheader("📸 Workspace Image Analysis")
st.markdown("Upload a workspace image for hazard detection analysis.")

uploaded_file = st.file_uploader(
    "Choose a workspace image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        frame = np.array(image.convert('RGB'))
        
        # Detect hazards
        hazard, red_pixels = detect_hazard(frame, hazard_threshold, confidence_threshold)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if hazard:
                st.error("🚨 HAZARD DETECTED")
                st.session_state['machine_status'] = "STOPPED"
                st.session_state['hazard_count'] += 1
            else:
                st.success("✅ WORKSPACE SAFE")
                st.session_state['machine_status'] = "Safe"
            
            st.write(f"**Red Pixels:** {red_pixels:,}")
            st.write(f"**Threshold:** {hazard_threshold:,}")
            st.write(f"**Machine Status:** {st.session_state['machine_status']}")
            st.write(f"**Hazard Events:** {st.session_state['hazard_count']}")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# System status
st.markdown("---")
st.markdown("### 🔧 System Status")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Machine Status", st.session_state['machine_status'])
with col2:
    st.metric("Hazard Events", st.session_state['hazard_count'])
with col3:
    status = "🔴 Stopped" if st.session_state['machine_status'] == "STOPPED" else "🟢 Safe"
    st.metric("System Health", status)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.9em;'>
    <p><strong>AI-Powered Accident Prevention System</strong></p>
    <p>International Automobile Centre of Excellence, Ahmedabad</p>
</div>
""", unsafe_allow_html=True) 