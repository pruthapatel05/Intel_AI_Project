# AI-Powered Accident Prevention in Modular Manufacturing Systems (MMS)

**Organisation:** International Automobile Centre of Excellence, Ahmedabad  
**Category:** Industry Defined Problem

---

## Executive Summary

Accidents in Modular Manufacturing Systems (MMS) often occur due to delayed human response or oversight. This project implements an AI-powered camera vision system that continuously monitors activities and can auto-stop machines in real-time upon detecting potential hazards. This proactive approach significantly reduces accidents, prevents injuries, and ensures workplace safety.

---

## Problem Statement

- **Delayed human response** and **oversight** are leading causes of accidents in MMS.
- Traditional safety systems are reactive, not proactive.
- There is a need for a real-time, automated, and reliable hazard detection and machine control system.

### Real-World Scenario

A static camera monitors a modular manufacturing system (MMS) floor in real time. A human worker enters the scene and walks dangerously close to an active robotic arm or conveyor belt. Some workers are wearing proper safety gear—helmet, vest, gloves, safety boots—while one worker is not wearing required PPE (e.g., missing helmet or vest). The robotic machine continues to operate until the AI system detects the hazard. The scene shows a clear near-miss accident caused by the unprotected worker's risky behavior in a high-activity zone.

**This is exactly the type of scenario our AI system is designed to prevent.**

## Solution

- **AI-powered camera vision** continuously monitors the workspace.
- **Real-time hazard detection** using advanced computer vision algorithms.
- **Automatic machine stop** when a hazard is detected, preventing accidents before they occur.
- **Ultra-strict detection logic** to eliminate false positives and negatives.

## Benefits

- **Reduces accidents and injuries** in industrial environments.
- **Ensures compliance** with safety standards.
- **Minimizes downtime** and production losses due to accidents.
- **Builds a culture of safety** and trust in the workplace.

---

## Features

- **Continuous real-time monitoring** via webcam or image upload.
- **Ultra-strict hazard detection** (minimizes false alarms).
- **Automatic machine status update** (Safe/Stopped) with clear visual indicators.
- **Hazard event counting** for safety analytics.
- **Professional, user-friendly interface** for industrial use.

---

## How It Works

1. **Camera feed** (webcam or uploaded image) is analyzed in real-time.
2. **AI vision algorithm** detects hazardous red objects/zones.
3. If a hazard is detected for a sustained period, the system **auto-stops the machine**.
4. **Status and hazard count** are displayed live on the dashboard.

---

## Getting Started

### 1. Install Python 3.7+

Download and install Python from [python.org](https://www.python.org/downloads/).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Option 1: With the launcher
```bash
python run_app.py
```
#### Option 2: Directly with Streamlit
```bash
streamlit run app.py
```

### 4. Access the Dashboard

Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## Usage

- **Webcam Mode:**
  - Select "Webcam" in the sidebar.
  - Click "Start Webcam" to begin monitoring.
  - The system will auto-stop the machine if a hazard is detected.

- **Image Upload Mode:**
  - Select "Image Upload" in the sidebar.
  - Upload a photo of the workspace.
  - The system will analyze the image for hazards.

- **Adjust Sensitivity:**
  - Use the sidebar sliders to fine-tune hazard detection for your environment.

---

## Industrial-Grade Detection Logic

- **High thresholds** for red object detection (minimizes false alarms).
- **Requires 10 consecutive hazard frames** to trigger a stop.
- **Requires 15 consecutive safe frames** to resume operation.
- **Minimum 1% of frame** must be hazardous for detection.
- **5-second cooldown** between hazard events.

---

## Troubleshooting

- **Webcam not detected:**
  - Ensure your camera is connected and not used by another app.
  - Try restarting your computer.

- **False positives/negatives:**
  - Adjust the sensitivity and confidence sliders in the sidebar.
  - Ensure good lighting and camera placement.

- **App not starting:**
  - Ensure all dependencies are installed.
  - Try running with `streamlit run app.py` for more detailed error output.

---

## For Developers

- **Code is fully commented** for maintainability.
- **Detection logic** can be extended for other hazard types (e.g., PPE detection, zone intrusion).
- **Model directory** is included for future AI model integration.

---

## License

This project is provided for educational and industrial research purposes at the International Automobile Centre of Excellence, Ahmedabad.

---

**Built with:** Streamlit, OpenCV, NumPy, Pillow  
**Last Updated:** 2024 