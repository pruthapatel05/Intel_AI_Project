"""
Kaggle Model Training Script for MMS Safety System
==================================================

This script downloads relevant datasets from Kaggle and trains custom models
for the AI-powered accident prevention system.

Usage:
    python train_kaggle_models.py --dataset ppe-detection --model-type yolo
    python train_kaggle_models.py --list-datasets
    python train_kaggle_models.py --download-only ppe-detection

Developed for: International Automobile Centre of Excellence, Ahmedabad
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaggle_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KaggleModelTrainer:
    """
    Trainer for downloading Kaggle datasets and training safety detection models.
    """
    
    def __init__(self):
        """Initialize the Kaggle model trainer."""
        self.datasets_config = self._load_datasets_config()
        self.base_dir = Path("kaggle_models")
        self.base_dir.mkdir(exist_ok=True)
        
    def _load_datasets_config(self):
        """Load datasets configuration from JSON file."""
        config_path = Path("model/kaggle_datasets.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.error("Kaggle datasets configuration file not found!")
            return {"safety_datasets": {}}
    
    def list_available_datasets(self):
        """List all available datasets for training."""
        logger.info("🏭 Available Kaggle Datasets for MMS Safety Training:")
        logger.info("=" * 60)
        
        datasets = self.datasets_config.get("safety_datasets", {})
        for key, dataset in datasets.items():
            logger.info(f"📊 Dataset: {key}")
            logger.info(f"   Name: {dataset['name']}")
            logger.info(f"   Description: {dataset['description']}")
            logger.info(f"   Type: {dataset['type']}")
            logger.info(f"   Format: {dataset['format']}")
            logger.info(f"   Classes: {', '.join(dataset['classes'])}")
            logger.info(f"   Images: {dataset['num_images']:,}")
            logger.info(f"   Tags: {', '.join(dataset['tags'])}")
            logger.info("-" * 40)
    
    def check_kaggle_credentials(self):
        """Check if Kaggle credentials are properly configured."""
        try:
            result = subprocess.run(['kaggle', 'datasets', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ Kaggle credentials are properly configured!")
                return True
            else:
                logger.error("❌ Kaggle credentials not found or invalid!")
                logger.error("Please configure Kaggle API credentials:")
                logger.error("1. Go to https://www.kaggle.com/settings/account")
                logger.error("2. Create API token")
                logger.error("3. Place kaggle.json in ~/.kaggle/ directory")
                return False
        except FileNotFoundError:
            logger.error("❌ Kaggle CLI not installed!")
            logger.error("Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking Kaggle credentials: {str(e)}")
            return False
    
    def download_dataset(self, dataset_key: str, download_only: bool = False):
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_key: Key of the dataset to download
            download_only: If True, only download without training
        """
        if not self.check_kaggle_credentials():
            return False
        
        datasets = self.datasets_config.get("safety_datasets", {})
        if dataset_key not in datasets:
            logger.error(f"❌ Dataset '{dataset_key}' not found!")
            self.list_available_datasets()
            return False
        
        dataset_info = datasets[dataset_key]
        dataset_name = dataset_info['name']
        
        logger.info(f"📥 Downloading dataset: {dataset_name}")
        logger.info(f"📋 Description: {dataset_info['description']}")
        logger.info(f"🏷️  Type: {dataset_info['type']}")
        logger.info(f"📁 Format: {dataset_info['format']}")
        
        # Create dataset directory
        dataset_dir = self.base_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download dataset using Kaggle CLI
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(dataset_dir), '--unzip']
            logger.info(f"🔄 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Dataset downloaded successfully to {dataset_dir}")
                
                # List downloaded files
                files = list(dataset_dir.rglob("*"))
                logger.info(f"📁 Downloaded {len(files)} files")
                
                if not download_only:
                    logger.info("🚀 Starting model training...")
                    return self.train_model(dataset_key, dataset_info)
                else:
                    logger.info("📥 Download completed. Use --train to start training.")
                    return True
            else:
                logger.error(f"❌ Error downloading dataset: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Download timed out. Dataset might be too large.")
            return False
        except Exception as e:
            logger.error(f"❌ Error during download: {str(e)}")
            return False
    
    def train_model(self, dataset_key: str, dataset_info: dict):
        """
        Train a model using the downloaded dataset.
        
        Args:
            dataset_key: Key of the dataset
            dataset_info: Dataset information
        """
        dataset_dir = self.base_dir / dataset_key
        model_type = dataset_info['format']
        
        logger.info(f"🎯 Training {model_type.upper()} model for {dataset_key}")
        logger.info(f"📁 Dataset directory: {dataset_dir}")
        
        try:
            # Import training module
            from model.train_safety_model import SafetyModelTrainer
            
            # Initialize trainer
            trainer = SafetyModelTrainer(output_dir=str(self.base_dir / "trained_models"))
            
            # Prepare dataset
            logger.info("🔧 Preparing dataset for training...")
            prepared_dataset = trainer.prepare_dataset(str(dataset_dir), model_type)
            
            # Train model
            logger.info("🏋️ Starting model training...")
            start_time = time.time()
            
            model_path = trainer.train_model(prepared_dataset, model_type)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
            logger.info(f"💾 Model saved to: {model_path}")
            
            # Show model information
            self._show_model_info(model_path, dataset_info)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during model training: {str(e)}")
            return False
    
    def _show_model_info(self, model_path: str, dataset_info: dict):
        """Display information about the trained model."""
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        logger.info("📊 Model Information:")
        logger.info(f"   Path: {model_path}")
        logger.info(f"   Size: {model_size:.2f} MB")
        logger.info(f"   Classes: {', '.join(dataset_info['classes'])}")
        logger.info(f"   Type: {dataset_info['type']}")
        logger.info(f"   Format: {dataset_info['format']}")
        
        # Show deployment recommendations
        self._show_deployment_recommendations(model_size, dataset_info)
    
    def _show_deployment_recommendations(self, model_size: float, dataset_info: dict):
        """Show deployment recommendations based on model size and type."""
        logger.info("🚀 Deployment Recommendations:")
        
        if model_size < 50:
            logger.info("   ✅ Suitable for edge devices")
            logger.info("   ✅ Real-time inference possible")
        elif model_size < 200:
            logger.info("   ✅ Suitable for server deployment")
            logger.info("   ⚠️  May need GPU for real-time performance")
        else:
            logger.info("   ⚠️  Large model - consider cloud deployment")
            logger.info("   ⚠️  GPU recommended for optimal performance")
        
        # Show integration instructions
        logger.info("🔧 Integration Instructions:")
        logger.info("   1. Copy the trained model to your project")
        logger.info("   2. Update app.py to use the new model")
        logger.info("   3. Test with your MMS environment")
        logger.info("   4. Deploy to production")
    
    def quick_start(self, dataset_key: str = "ppe-detection"):
        """
        Quick start function for common use case.
        
        Args:
            dataset_key: Dataset to use for quick start
        """
        logger.info("🚀 Quick Start: Training PPE Detection Model")
        logger.info("=" * 50)
        
        # Check if dataset already exists
        dataset_dir = self.base_dir / dataset_key
        if dataset_dir.exists():
            logger.info(f"📁 Dataset already exists at {dataset_dir}")
            logger.info("🔄 Starting training with existing dataset...")
            datasets = self.datasets_config.get("safety_datasets", {})
            if dataset_key in datasets:
                return self.train_model(dataset_key, datasets[dataset_key])
        else:
            logger.info("📥 Downloading dataset first...")
            return self.download_dataset(dataset_key, download_only=False)
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        logger.info("🧹 Cleaning up temporary files...")
        
        # Remove temporary files
        temp_patterns = ["*.tmp", "*.temp", "__pycache__", "*.pyc"]
        for pattern in temp_patterns:
            for file_path in self.base_dir.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
        
        logger.info("✅ Cleanup completed")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Train safety detection models using Kaggle datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_kaggle_models.py --list-datasets
  python train_kaggle_models.py --dataset ppe-detection --model-type yolo
  python train_kaggle_models.py --download-only ppe-detection
  python train_kaggle_models.py --quick-start
  python train_kaggle_models.py --cleanup
        """
    )
    
    parser.add_argument("--list-datasets", action="store_true",
                       help="List all available datasets")
    parser.add_argument("--dataset", type=str,
                       help="Dataset key to download and train")
    parser.add_argument("--model-type", choices=["yolo", "tensorflow", "pytorch"],
                       default="yolo", help="Type of model to train")
    parser.add_argument("--download-only", type=str,
                       help="Only download dataset without training")
    parser.add_argument("--quick-start", action="store_true",
                       help="Quick start with PPE detection dataset")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = KaggleModelTrainer()
    
    try:
        if args.list_datasets:
            trainer.list_available_datasets()
            
        elif args.download_only:
            trainer.download_dataset(args.download_only, download_only=True)
            
        elif args.dataset:
            trainer.download_dataset(args.dataset, download_only=False)
            
        elif args.quick_start:
            trainer.quick_start()
            
        elif args.cleanup:
            trainer.cleanup()
            
        else:
            logger.info("🏭 MMS Safety Model Training System")
            logger.info("=" * 40)
            logger.info("Use --list-datasets to see available datasets")
            logger.info("Use --quick-start for immediate training")
            logger.info("Use --help for more options")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 