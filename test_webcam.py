import cv2
import numpy as np

def test_webcam():
    """Test webcam functionality"""
    print("🔍 Testing webcam functionality...")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        print("Possible issues:")
        print("- Camera not connected")
        print("- Camera permissions not granted")
        print("- Camera being used by another application")
        return False
    
    print("✅ Webcam opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read frame from webcam")
        cap.release()
        return False
    
    print(f"✅ Frame read successfully - Shape: {frame.shape}")
    print(f"   Width: {frame.shape[1]}, Height: {frame.shape[0]}")
    
    # Set some properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam properties set")
    
    # Test reading multiple frames
    print("📹 Testing frame capture...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"   Frame {i+1}: ✅ Success")
        else:
            print(f"   Frame {i+1}: ❌ Failed")
    
    cap.release()
    print("✅ Webcam test completed successfully")
    return True

if __name__ == "__main__":
    test_webcam() 