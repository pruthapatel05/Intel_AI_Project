"""
Safety Model Training Script
============================

This script provides functionality to train custom safety detection models
for the MMS Safety System.

Developed for: International Automobile Centre of Excellence, Ahmedabad
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ppe_detection_model(dataset_path: str, output_path: str = "models"):
    """
    Train a PPE detection model using the provided dataset.
    
    Args:
        dataset_path: Path to the PPE dataset
        output_path: Directory to save the trained model
    """
    try:
        from ultralytics import YOLO
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        logger.info(f"Starting PPE detection model training with dataset: {dataset_path}")
        results = model.train(
            data=dataset_path,
            epochs=100,
            imgsz=640,
            batch=16,
            name='ppe_detection_model',
            project=output_path
        )
        
        logger.info("PPE detection model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error training PPE detection model: {str(e)}")
        return False

def train_hazard_classification_model(dataset_path: str, output_path: str = "models"):
    """
    Train a hazard classification model using the provided dataset.
    
    Args:
        dataset_path: Path to the hazard classification dataset
        output_path: Directory to save the trained model
    """
    try:
        import torch
        import torchvision
        from torchvision import transforms
        from torch.utils.data import DataLoader, ImageFolder
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        dataset = ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model (ResNet18 for classification)
        model = torchvision.models.resnet18(pretrained=True)
        num_classes = len(dataset.classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        logger.info(f"Starting hazard classification model training with {num_classes} classes")
        model.train()
        
        for epoch in range(10):  # 10 epochs for demonstration
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if i % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        # Save the model
        model_path = os.path.join(output_path, "hazard_classification_model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Hazard classification model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training hazard classification model: {str(e)}")
        return False

def main():
    """Main function to run model training."""
    print("🏭 MMS Safety Model Training")
    print("=" * 50)
    
    # Check if dataset paths are provided
    if len(sys.argv) < 2:
        print("Usage: python train_safety_model.py <ppe_dataset_path> [hazard_dataset_path]")
        print("Example: python train_safety_model.py ./datasets/ppe ./datasets/hazards")
        return
    
    ppe_dataset_path = sys.argv[1]
    hazard_dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Train PPE detection model
    if os.path.exists(ppe_dataset_path):
        print(f"\n📦 Training PPE Detection Model...")
        success = train_ppe_detection_model(ppe_dataset_path)
        if success:
            print("✅ PPE detection model training completed!")
        else:
            print("❌ PPE detection model training failed!")
    else:
        print(f"❌ PPE dataset not found at: {ppe_dataset_path}")
    
    # Train hazard classification model
    if hazard_dataset_path and os.path.exists(hazard_dataset_path):
        print(f"\n📦 Training Hazard Classification Model...")
        success = train_hazard_classification_model(hazard_dataset_path)
        if success:
            print("✅ Hazard classification model training completed!")
        else:
            print("❌ Hazard classification model training failed!")
    elif hazard_dataset_path:
        print(f"❌ Hazard dataset not found at: {hazard_dataset_path}")
    
    print("\n🎯 Training process completed!")

if __name__ == "__main__":
    main() 