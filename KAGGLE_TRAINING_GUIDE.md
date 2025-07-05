# 🏭 Kaggle Model Training Guide for MMS Safety System

**Organisation:** International Automobile Centre of Excellence, Ahmedabad  
**Category:** Industry Defined Problem - Modular Manufacturing Systems Safety

---

## 🎯 Overview

This guide explains how to train custom AI models using Kaggle datasets for the MMS Safety System. The system supports multiple pre-trained models for comprehensive safety monitoring including PPE detection, person detection, zone intrusion, and hazard classification.

---

## 📊 Available Kaggle Datasets

### 1. **PPE Detection Dataset**
- **Kaggle:** `gti-upm/ppe-detection-dataset`
- **Classes:** helmet, vest, gloves, boots
- **Images:** 5,000
- **Type:** Object Detection
- **Format:** YOLO

### 2. **Safety Equipment Detection**
- **Kaggle:** `andrewmvd/safety-equipment-detection`
- **Classes:** hard_hat, safety_vest, safety_glasses, ear_protection
- **Images:** 3,000
- **Type:** Object Detection
- **Format:** YOLO

### 3. **Industrial Hazards Dataset**
- **Kaggle:** `mohamedhanytawfeek/industrial-hazards-dataset`
- **Classes:** safe, hazard, warning, emergency
- **Images:** 2,000
- **Type:** Classification
- **Format:** TensorFlow

### 4. **Person Detection Dataset**
- **Kaggle:** `crowdai/crowdai-human-detection-challenge`
- **Classes:** person
- **Images:** 8,000
- **Type:** Object Detection
- **Format:** YOLO

### 5. **Construction Safety Dataset**
- **Kaggle:** `andrewmvd/construction-site-safety`
- **Classes:** worker, helmet, vest, machinery, danger_zone
- **Images:** 4,000
- **Type:** Object Detection
- **Format:** YOLO

---

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure Kaggle API:**
   - Go to [Kaggle Settings](https://www.kaggle.com/settings/account)
   - Create API token
   - Download `kaggle.json`
   - Place in `~/.kaggle/` directory (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

3. **Verify Installation:**
```bash
kaggle datasets list
```

### Quick Training

```bash
# List available datasets
python train_kaggle_models.py --list-datasets

# Quick start with PPE detection
python train_kaggle_models.py --quick-start

# Train specific dataset
python train_kaggle_models.py --dataset ppe-detection --model-type yolo

# Download only (without training)
python train_kaggle_models.py --download-only ppe-detection
```

---

## 📋 Detailed Training Process

### Step 1: Dataset Selection

Choose the appropriate dataset based on your requirements:

| Use Case | Recommended Dataset | Model Type |
|----------|-------------------|------------|
| PPE Compliance | `ppe-detection` | YOLO |
| Safety Equipment | `safety-equipment` | YOLO |
| Hazard Classification | `industrial-hazards` | TensorFlow |
| Person Monitoring | `person-detection` | YOLO |
| Construction Safety | `construction-safety` | YOLO |

### Step 2: Download Dataset

```bash
python train_kaggle_models.py --dataset ppe-detection --download-only
```

This will:
- Download the dataset from Kaggle
- Extract files to `kaggle_models/ppe-detection/`
- Create necessary configuration files

### Step 3: Train Model

```bash
python train_kaggle_models.py --dataset ppe-detection --model-type yolo
```

Training process includes:
- Dataset preparation and validation
- Model initialization with pre-trained weights
- Training with data augmentation
- Validation and model selection
- Model export and optimization

### Step 4: Model Integration

After training, integrate the model into your MMS system:

```python
# In app.py
from model.safety_detection_model import SafetyDetectionModel

# Load trained model
model = SafetyDetectionModel("kaggle_models/trained_models/safety_detection_yolo.pt")

# Use for detection
results = model.comprehensive_safety_analysis(frame)
```

---

## 🎛️ Training Configurations

### YOLO Training Configuration

```yaml
base_model: yolov8n.pt
epochs: 100
batch_size: 16
image_size: 640
learning_rate: 0.01
patience: 20
augmentation: true
pretrained: true
```

### TensorFlow Training Configuration

```yaml
base_model: resnet50
epochs: 50
batch_size: 32
image_size: 224
learning_rate: 0.001
validation_split: 0.2
augmentation: true
pretrained: true
```

### Custom Configuration

You can override default settings:

```bash
python train_kaggle_models.py --dataset ppe-detection --model-type yolo \
    --epochs 150 --batch-size 32 --learning-rate 0.005
```

---

## 📈 Model Performance Requirements

### PPE Detection
- **Minimum Accuracy:** 85%
- **Minimum Precision:** 80%
- **Minimum Recall:** 85%
- **Target FPS:** 30

### Hazard Detection
- **Minimum Accuracy:** 90%
- **Minimum Precision:** 85%
- **Minimum Recall:** 90%
- **Target FPS:** 25

### Person Detection
- **Minimum Accuracy:** 95%
- **Minimum Precision:** 90%
- **Minimum Recall:** 95%
- **Target FPS:** 30

---

## 🚀 Deployment Options

### Edge Device Deployment
- **Model Size:** < 50MB
- **Inference Time:** < 100ms
- **Memory Usage:** < 2GB
- **CPU Cores:** 2+
- **GPU:** Optional

### Server Deployment
- **Model Size:** < 200MB
- **Inference Time:** < 50ms
- **Memory Usage:** < 8GB
- **CPU Cores:** 4+
- **GPU:** Recommended

### Cloud Deployment
- **Model Size:** < 500MB
- **Inference Time:** < 25ms
- **Memory Usage:** < 16GB
- **CPU Cores:** 8+
- **GPU:** Required

---

## 🔧 Advanced Training Options

### Custom Dataset Preparation

1. **Organize your data:**
```
custom_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

2. **Create data.yaml:**
```yaml
path: /path/to/custom_dataset
train: train/images
val: val/images
nc: 4  # number of classes
names: ['helmet', 'vest', 'gloves', 'boots']
```

3. **Train with custom dataset:**
```bash
python model/train_safety_model.py --dataset-path custom_dataset --model-type yolo
```

### Transfer Learning

For better performance with limited data:

```python
# Load pre-trained model
model = SafetyDetectionModel("pretrained_model.pt")

# Fine-tune on your dataset
model.train_custom_model("your_dataset", "yolo")
```

### Data Augmentation

Automatic augmentation includes:
- Random rotation (±15°)
- Random scaling (0.8-1.2x)
- Random brightness (±20%)
- Random contrast (±20%)
- Random horizontal flip
- Random noise injection

---

## 📊 Model Evaluation

### Evaluation Metrics

After training, models are evaluated on:

1. **Accuracy:** Overall classification accuracy
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Detailed classification results

### Performance Testing

```bash
# Evaluate trained model
python model/train_safety_model.py --evaluate model_path test_dataset_path

# Benchmark performance
python model/benchmark_model.py --model model_path --dataset test_dataset
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **Kaggle API Error:**
   ```
   ❌ Kaggle credentials not found or invalid!
   ```
   **Solution:** Verify kaggle.json is in the correct location and has proper permissions.

2. **CUDA Out of Memory:**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Reduce batch size or use CPU training.

3. **Dataset Download Timeout:**
   ```
   ❌ Download timed out. Dataset might be too large.
   ```
   **Solution:** Use `--download-only` and retry, or download manually.

4. **Model Training Fails:**
   ```
   ❌ Error during model training
   ```
   **Solution:** Check dataset format and ensure all dependencies are installed.

### Performance Optimization

1. **For Faster Training:**
   - Use GPU acceleration
   - Reduce image size
   - Increase batch size
   - Use mixed precision training

2. **For Better Accuracy:**
   - Increase training epochs
   - Use data augmentation
   - Fine-tune learning rate
   - Use larger base model

3. **For Deployment:**
   - Model quantization
   - TensorRT optimization
   - ONNX conversion
   - Edge device optimization

---

## 📚 Additional Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TensorFlow Training Guide](https://www.tensorflow.org/tutorials)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

### Datasets
- [Kaggle Safety Datasets](https://www.kaggle.com/datasets?search=safety)
- [Roboflow Safety Datasets](https://universe.roboflow.com/search?q=safety)
- [Open Images Safety](https://storage.googleapis.com/openimages/web/index.html)

### Papers
- "PPE Detection in Industrial Environments" - IEEE Safety Conference 2023
- "Real-time Safety Monitoring in Manufacturing" - IJCAI 2023
- "AI-powered Accident Prevention" - Safety Science Journal 2023

---

## 🤝 Support

For technical support or questions:

1. **Check the logs:** `kaggle_training.log`
2. **Review documentation:** This guide and inline code comments
3. **Test with sample data:** Use provided test datasets
4. **Contact support:** Technical team at International Automobile Centre of Excellence

---

**Built with:** Python, TensorFlow, PyTorch, YOLOv8, Kaggle API  
**Last Updated:** 2024  
**Version:** 1.0.0 