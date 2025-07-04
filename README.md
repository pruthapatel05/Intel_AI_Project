# AI-Based Accident Prevention in MMS

**Organisation:** International Automobile Centre of Excellence, Ahmedabad  
**Category:** Industry Defined Problem

## Description
This project implements a modern, production-quality web application for real-time accident prevention in Modular Manufacturing Systems (MMS) using AI-powered camera vision. The system continuously monitors activities, detects hazards, and simulates machine auto-stop to ensure workplace safety.

## Features
- Real-time camera feed with AI hazard detection
- Modern, responsive web dashboard (Flask + Bootstrap)
- Sidebar for configuration (hazard sensitivity, input source)
- Live status updates and overlays
- Easily swappable AI model (default: red object detection)

## Project Structure
```
├── main.py              # Flask backend (video, AI, status)
├── requirements.txt     # Python dependencies
├── static/              # CSS, JS, and static assets
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html       # Main dashboard UI
└── README.md            # Project documentation
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**
   ```bash
   python main.py
   ```
4. **Open the dashboard**
   Go to [http://localhost:5000](http://localhost:5000) in your browser.

## Notes
- The AI model is a placeholder; you can replace it with your own hazard detection model.
- The machine auto-stop is simulated in code for demonstration purposes.

## Requirements
- Python 3.7+
- Flask
- OpenCV
- TensorFlow
- numpy
- Pillow

---
For more details, see the code comments and documentation in each file. 