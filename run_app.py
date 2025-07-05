#!/usr/bin/env python3
"""
AI-Based Accident Prevention System - Startup Script
This script provides a reliable way to launch the application with proper error handling.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_webcam():
    """Check if webcam is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✅ Webcam is available")
                return True
            else:
                print("⚠️  Webcam detected but not responding properly")
                return False
        else:
            print("⚠️  Webcam not available (will use image upload mode)")
            return False
    except Exception as e:
        print(f"⚠️  Could not check webcam: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit application."""
    print("\n🚀 Launching AI-Based Accident Prevention System...")
    print("=" * 60)
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    try:
        # Launch Streamlit
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
        
        print("⏳ Starting server...")
        time.sleep(3)  # Wait for server to start
        
        # Try to open browser
        try:
            webbrowser.open('http://localhost:8501')
            print("🌐 Opening application in your default browser...")
        except Exception:
            print("🌐 Please open your browser and go to: http://localhost:8501")
        
        print("\n📋 Application Information:")
        print("   • URL: http://localhost:8501")
        print("   • Press Ctrl+C to stop the application")
        print("   • Use the sidebar to configure detection settings")
        print("   • Switch between webcam and image upload modes")
        
        print("\n" + "=" * 60)
        print("🎯 Application is running! Check your browser.")
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping application...")
            process.terminate()
            process.wait()
            print("✅ Application stopped successfully!")
            
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        return False
    
    return True

def main():
    """Main function to run the startup process."""
    print("🤖 AI-Based Accident Prevention System")
    print("=" * 60)
    
    # Launch the application
    return launch_streamlit()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Failed to start the application.")
            print("💡 Try running: python -m streamlit run app.py")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 