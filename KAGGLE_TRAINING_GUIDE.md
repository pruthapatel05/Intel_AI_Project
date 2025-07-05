# 🏭 Kaggle Training Guide for MMS Safety System

## Overview

This guide explains how to train custom AI models using Kaggle datasets for the MMS Safety System.

## Prerequisites

### 1. Kaggle API Setup
1. Go to [kaggle.com](https://www.kaggle.com)
2. Sign in to your account
3. Go to "Account" → "API" → "Create New API Token"
4. Download `kaggle.json`
5. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

### 2. Python Dependencies
```bash
pip install kaggle ultralytics torch torchvision transformers
```

## Available Datasets

### 1. Construction Safety Dataset
- **Purpose**: PPE detection in construction environments
- **Classes**: helmet, vest, gloves, boots, person
- **Type**: Object Detection
- **Format**: YOLO

### 2. PPE Detection Dataset
- **Purpose**: Personal Protective Equipment detection
- **Classes**: hardhat, vest, gloves, boots, goggles
- **Type**: Object Detection
- **Format**: YOLO

## Training Process

### Step 1: Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd Intel_AI

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Datasets
Edit `model/kaggle_datasets.json` to include your desired datasets:

```json
{
  "datasets": [
    {
      "name": "your-dataset-name",
      "description": "Description of your dataset",
      "url": "https://www.kaggle.com/datasets/your-dataset",
      "type": "object_detection",
      "classes": ["class1", "class2", "class3"]
    }
  ]
}
```

### Step 3: Run Training
```bash
# Train all configured models
python train_kaggle_models.py

# Train specific model
python model/train_safety_model.py <dataset_path>
```

## Training Configuration

### YOLO Model Parameters
- **Epochs**: 100 (default)
- **Image Size**: 640x640
- **Batch Size**: 16
- **Learning Rate**: Auto (YOLOv8 default)

### Custom Training
You can modify training parameters in `train_kaggle_models.py`:

```python
results = model.train(
    data=dataset_path,
    epochs=100,          # Number of epochs
    imgsz=640,          # Image size
    batch=16,           # Batch size
    name=model_name,    # Model name
    project="kaggle_models"  # Project directory
)
```

## Model Output

Trained models are saved in:
- `kaggle_models/` - Main project directory
- `kaggle_models/<model_name>/weights/best.pt` - Best model weights
- `kaggle_models/<model_name>/weights/last.pt` - Last checkpoint

## Integration with Main App

### 1. Update Model Path
In `app.py`, update the model path:

```python
# Load custom trained model
model_path = "kaggle_models/construction-safety_model/weights/best.pt"
safety_model = SafetyDetectionModel(model_path)
```

### 2. Test Model
```python
# Test the trained model
results = safety_model.comprehensive_safety_analysis(frame)
print(f"Risk Level: {results['overall_risk_level']}")
```

## Troubleshooting

### Common Issues

1. **Kaggle API Error**
   ```
   Error: Could not find kaggle.json
   ```
   **Solution**: Ensure `kaggle.json` is in the correct location

2. **Dataset Download Error**
   ```
   Error: Dataset not found
   ```
   **Solution**: Check dataset name in `kaggle_datasets.json`

3. **Training Memory Error**
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce batch size or image size

4. **Model Loading Error**
   ```
   Error: Model file not found
   ```
   **Solution**: Check if training completed successfully

### Performance Optimization

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Reduce Epochs**: Start with 10-20 epochs for testing
3. **Smaller Images**: Use 416x416 for faster training
4. **Data Augmentation**: Enable for better generalization

## Best Practices

1. **Dataset Quality**: Ensure high-quality, well-labeled data
2. **Validation**: Use separate validation set
3. **Monitoring**: Check training logs for overfitting
4. **Backup**: Keep multiple model checkpoints
5. **Testing**: Test models on real-world data

## Advanced Features

### Custom Loss Functions
```python
# Implement custom loss in train_safety_model.py
def custom_loss(predictions, targets):
    # Your custom loss implementation
    return loss
```

### Multi-GPU Training
```python
# Enable multi-GPU training
model.train(
    data=dataset_path,
    epochs=100,
    device=[0, 1]  # Use GPUs 0 and 1
)
```

### Transfer Learning
```python
# Load pre-trained weights
model = YOLO('yolov8l.pt')  # Use larger model
model.train(data=dataset_path, epochs=50)
```

## Monitoring and Logging

Training progress is logged in:
- `kaggle_training.log` - Main training log
- `kaggle_models/<model_name>/results.csv` - Training metrics
- `kaggle_models/<model_name>/confusion_matrix.png` - Model performance

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review training logs
3. Verify dataset format
4. Test with smaller datasets first

---

**Developed for**: International Automobile Centre of Excellence, Ahmedabad  
**Category**: Industry Defined Problem - MMS Safety 