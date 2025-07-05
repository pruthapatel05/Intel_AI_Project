"""
Kaggle Models Training Script
============================

This script downloads and trains models using Kaggle datasets for the MMS Safety System.

Developed for: International Automobile Centre of Excellence, Ahmedabad
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    try:
        import kaggle
        
        # Check if kaggle.json exists
        kaggle_config_path = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config_path.exists():
            logger.warning("Kaggle credentials not found. Please setup kaggle.json")
            return False
        
        # Authenticate
        kaggle.api.authenticate()
        logger.info("Kaggle API authenticated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle credentials: {str(e)}")
        return False

def download_kaggle_dataset(dataset_name: str, target_dir: str = "datasets"):
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Name of the Kaggle dataset
        target_dir: Directory to save the dataset
    """
    try:
        import kaggle
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Download dataset
        logger.info(f"Downloading dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        
        logger.info(f"Dataset {dataset_name} downloaded successfully to {target_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
        return False

def train_yolo_model(dataset_path: str, model_name: str, epochs: int = 100):
    """
    Train a YOLO model on the provided dataset.
    
    Args:
        dataset_path: Path to the dataset
        model_name: Name for the trained model
        epochs: Number of training epochs
    """
    try:
        from ultralytics import YOLO
        
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        logger.info(f"Training YOLO model: {model_name}")
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=model_name,
            project="kaggle_models"
        )
        
        logger.info(f"YOLO model {model_name} training completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error training YOLO model {model_name}: {str(e)}")
        return False

def load_dataset_config():
    """Load dataset configuration from kaggle_datasets.json."""
    try:
        config_path = Path("model/kaggle_datasets.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("kaggle_datasets.json not found")
            return {"datasets": []}
    except Exception as e:
        logger.error(f"Error loading dataset config: {str(e)}")
        return {"datasets": []}

def main():
    """Main function to run Kaggle model training."""
    print("🏭 Kaggle Models Training for MMS Safety System")
    print("=" * 60)
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        print("❌ Failed to setup Kaggle credentials")
        print("Please ensure kaggle.json is properly configured")
        return
    
    # Load dataset configuration
    config = load_dataset_config()
    
    if not config.get("datasets"):
        print("❌ No datasets configured in kaggle_datasets.json")
        return
    
    print(f"\n📦 Found {len(config['datasets'])} datasets to process")
    
    # Process each dataset
    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        dataset_type = dataset["type"]
        
        print(f"\n🔍 Processing dataset: {dataset_name}")
        print(f"   Type: {dataset_type}")
        print(f"   Classes: {', '.join(dataset['classes'])}")
        
        # Download dataset
        if download_kaggle_dataset(dataset_name):
            # Train model
            dataset_path = f"datasets/{dataset_name}"
            if os.path.exists(dataset_path):
                model_name = f"{dataset_name}_model"
                if train_yolo_model(dataset_path, model_name):
                    print(f"✅ Successfully trained model for {dataset_name}")
                else:
                    print(f"❌ Failed to train model for {dataset_name}")
            else:
                print(f"❌ Dataset directory not found: {dataset_path}")
        else:
            print(f"❌ Failed to download dataset: {dataset_name}")
    
    print("\n🎯 Kaggle models training process completed!")

if __name__ == "__main__":
    main() 