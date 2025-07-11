# 🏭 MMS Safety System - AI-Powered Accident Prevention

This project implements a real-time AI vision system for detecting hazards in industrial manufacturing environments. The system continuously monitors workspace activities and can automatically stop machines when potential hazards are detected, preventing accidents and ensuring workplace safety.

## 🚀 Features
- Real-time hazard detection using computer vision (YOLOv8)
- Ultra-strict detection logic to minimize false alarms
- Automatic machine status control (Safe/Stopped)
- Professional, modern Streamlit interface (dark theme by default)
- Supports webcam, image upload, and video upload
- Configurable sensitivity and confidence thresholds
- Industrial-grade status and metric cards

## 📦 Project Structure
```
Intel_AI/
├── streamlit_app.py         # Main Streamlit app
├── app.py                   # (Optional) Additional logic/functions
├── requirements.txt         # Python dependencies
├── .streamlit/config.toml   # Streamlit theme configuration
├── model weights (best.pt)  # Trained YOLOv8 weights
├── kaggle_models/           # (Optional) Dataset for retraining
└── README.md                # This file
```

## 🛠️ Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Intel_AI
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your trained YOLOv8 weights:**
   - Place your `best.pt` file in the correct path (update `MODEL_PATH` in `streamlit_app.py` if needed).

4. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🖥️ Usage
- **Webcam Mode:**
  - Select "Webcam" in the sidebar and click "Start Real-Time Monitoring".
- **Image Upload:**
  - Select "Image Upload" and upload a photo for hazard detection.
- **Video Upload:**
  - Select "Video Upload" and upload a video file for frame-by-frame analysis.
- **Adjust Sensitivity/Confidence:**
  - Use the sidebar sliders to fine-tune detection for your environment.

## ⚙️ Configuration
- The app uses a dark theme with red accents by default (see `.streamlit/config.toml`).
- You can adjust the model path and other settings in `streamlit_app.py`.

## 🤝 Credits
Developed for: International Automobile Centre of Excellence, Ahmedabad

Built with: Streamlit, Ultralytics YOLOv8, OpenCV, NumPy, Pillow

---

**For questions or contributions, please open an issue or pull request!** 