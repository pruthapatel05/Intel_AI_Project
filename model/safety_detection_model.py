"""
Advanced AI Safety Detection Model for MMS
==========================================

This module integrates multiple pre-trained models for comprehensive safety monitoring:
1. YOLOv8 for person detection and PPE classification
2. Custom trained models for safety equipment detection
3. Zone intrusion detection algorithms
4. Hazard classification and risk assessment

Developed for: International Automobile Centre of Excellence, Ahmedabad
"""

import numpy as np
import torch
from ultralytics import YOLO
import tensorflow as tf
from transformers import pipeline
import os
import tempfile
import requests
from typing import Dict, List, Tuple, Optional
import logging

# Handle OpenCV import with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Some features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyDetectionModel:
    """
    Advanced AI model for comprehensive safety monitoring in MMS environments.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the safety detection model with pre-trained weights.
        
        Args:
            model_path: Path to custom trained model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.person_detector = None
        self.ppe_detector = None
        self.hazard_classifier = None
        self.zone_detector = None
        
        # Model configuration
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Safety zones configuration
        self.machine_zone = None
        self.restricted_areas = []
        
        # Load models
        self._load_models(model_path)
        
    def _load_models(self, model_path: str = None):
        """Load all required models for safety detection."""
        try:
            # Load YOLOv8 for person detection
            logger.info("Loading YOLOv8 person detection model...")
            self.person_detector = YOLO('yolov8n.pt')
            
            # Load PPE detection model (custom or pre-trained)
            logger.info("Loading PPE detection model...")
            if model_path and os.path.exists(model_path):
                self.ppe_detector = YOLO(model_path)
            else:
                # Use a general object detection model for PPE
                self.ppe_detector = YOLO('yolov8n.pt')
            
            # Load hazard classifier
            logger.info("Loading hazard classification model...")
            self._load_hazard_classifier()
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_hazard_classifier(self):
        """Load hazard classification model."""
        try:
            # Use a pre-trained image classification model
            self.hazard_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load hazard classifier: {str(e)}")
            self.hazard_classifier = None
    
    def set_machine_zone(self, x1: int, y1: int, x2: int, y2: int):
        """
        Set the machine zone coordinates for intrusion detection.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
        """
        self.machine_zone = (x1, y1, x2, y2)
        logger.info(f"Machine zone set to: {self.machine_zone}")
    
    def add_restricted_area(self, x1: int, y1: int, x2: int, y2: int, area_type: str = "danger"):
        """
        Add a restricted area for enhanced safety monitoring.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            area_type: Type of restricted area ("danger", "warning", "caution")
        """
        self.restricted_areas.append({
            'coords': (x1, y1, x2, y2),
            'type': area_type
        })
        logger.info(f"Added restricted area: {area_type} at {x1},{y1},{x2},{y2}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in the frame using YOLOv8.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected persons with bounding boxes and confidence scores
        """
        try:
            if self.person_detector is None:
                return []
            
            # Run person detection
            results = self.person_detector(frame, classes=[0])  # class 0 is person in COCO
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf[0] > self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            persons.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence),
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            })
            
            return persons
            
        except Exception as e:
            logger.error(f"Error in person detection: {str(e)}")
            return []
    
    def detect_ppe(self, frame: np.ndarray, person_bbox: Tuple) -> Dict:
        """
        Detect Personal Protective Equipment (PPE) for a detected person.
        
        Args:
            frame: Input image frame
            person_bbox: Bounding box of detected person (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing PPE detection results
        """
        try:
            x1, y1, x2, y2 = person_bbox
            
            # Extract person region
            person_region = frame[y1:y2, x1:x2]
            if person_region.size == 0:
                return {'helmet': False, 'vest': False, 'gloves': False, 'boots': False}
            
            # Run PPE detection on person region
            results = self.ppe_detector(person_region)
            
            ppe_status = {
                'helmet': False,
                'vest': False,
                'gloves': False,
                'boots': False
            }
            
            # Analyze detection results for PPE items
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf[0] > self.confidence_threshold:
                            class_id = int(box.cls[0].cpu().numpy())
                            # Map class IDs to PPE items (this would need to be customized based on your dataset)
                            if class_id == 0:  # Assuming class 0 is helmet
                                ppe_status['helmet'] = True
                            elif class_id == 1:  # Assuming class 1 is vest
                                ppe_status['vest'] = True
                            elif class_id == 2:  # Assuming class 2 is gloves
                                ppe_status['gloves'] = True
                            elif class_id == 3:  # Assuming class 3 is boots
                                ppe_status['boots'] = True
            
            return ppe_status
            
        except Exception as e:
            logger.error(f"Error in PPE detection: {str(e)}")
            return {'helmet': False, 'vest': False, 'gloves': False, 'boots': False}
    
    def check_zone_intrusion(self, person_center: Tuple) -> Dict:
        """
        Check if a person is intruding into restricted zones.
        
        Args:
            person_center: Center coordinates of detected person (x, y)
            
        Returns:
            Dictionary containing intrusion status
        """
        intrusion_status = {
            'machine_zone': False,
            'restricted_areas': []
        }
        
        px, py = person_center
        
        # Check machine zone intrusion
        if self.machine_zone:
            x1, y1, x2, y2 = self.machine_zone
            if x1 <= px <= x2 and y1 <= py <= y2:
                intrusion_status['machine_zone'] = True
        
        # Check restricted areas intrusion
        for area in self.restricted_areas:
            x1, y1, x2, y2 = area['coords']
            if x1 <= px <= x2 and y1 <= py <= y2:
                intrusion_status['restricted_areas'].append({
                    'type': area['type'],
                    'coords': area['coords']
                })
        
        return intrusion_status
    
    def classify_hazard(self, frame: np.ndarray) -> Dict:
        """
        Classify potential hazards in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing hazard classification results
        """
        try:
            if self.hazard_classifier is None:
                return {'hazard_type': 'unknown', 'confidence': 0.0}
            
            # Convert BGR to RGB
            if OPENCV_AVAILABLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Fallback: assume frame is already RGB or convert using numpy
                rgb_frame = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Run classification
            results = self.hazard_classifier(pil_image)
            
            # Process results (this would need to be customized based on your hazard classes)
            if results:
                top_result = results[0]
                return {
                    'hazard_type': top_result['label'],
                    'confidence': top_result['score']
                }
            
            return {'hazard_type': 'unknown', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in hazard classification: {str(e)}")
            return {'hazard_type': 'unknown', 'confidence': 0.0}
    
    def detect_red_hazards(self, frame: np.ndarray, threshold: int = 15000) -> Tuple[bool, np.ndarray, int]:
        """
        Enhanced red hazard detection with improved algorithms.
        
        Args:
            frame: Input image frame (BGR format)
            threshold: Minimum red pixel count for hazard detection
            
        Returns:
            Tuple: (hazard_detected, red_mask, red_pixel_count)
        """
        if not OPENCV_AVAILABLE:
            # Fallback to basic numpy-based detection
            return self._detect_red_hazards_numpy(frame, threshold)
        
        try:
            if frame is None or frame.size == 0:
                return False, np.zeros((1, 1), dtype=np.uint8), 0
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Enhanced red detection with multiple ranges
            # Lower red range (0-10 degrees)
            lower_red1 = np.array([0, 150, 100])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            # Upper red range (170-180 degrees)
            lower_red2 = np.array([170, 150, 100])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Combine masks
            red_mask = mask1 + mask2
            
            # Apply morphological operations
            kernel = np.ones((7,7), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # Count red pixels
            red_pixels = cv2.countNonZero(red_mask)
            total_pixels = frame.shape[0] * frame.shape[1]
            red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
            
            # Enhanced hazard detection criteria
            min_red_percentage = 0.01
            hazard = (
                red_pixels > threshold * 2 and
                red_percentage > min_red_percentage and
                red_percentage > 0.02
            )
            
            return hazard, red_mask, red_pixels
            
        except Exception as e:
            logger.error(f"Error in red hazard detection: {str(e)}")
            return False, np.zeros((1, 1), dtype=np.uint8), 0
    
    def _detect_red_hazards_numpy(self, frame: np.ndarray, threshold: int = 15000) -> Tuple[bool, np.ndarray, int]:
        """
        Fallback red hazard detection using numpy (when OpenCV is not available).
        """
        try:
            if frame is None or frame.size == 0:
                return False, np.zeros((1, 1), dtype=np.uint8), 0
            
            # Simple red pixel detection using numpy
            red_pixels = np.sum((frame[:, :, 2] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 0] < 100))
            total_pixels = frame.shape[0] * frame.shape[1]
            red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
            
            # Create a simple mask (all zeros for now)
            red_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            hazard = red_pixels > threshold and red_percentage > 0.01
            return hazard, red_mask, red_pixels
            
        except Exception as e:
            logger.error(f"Error in numpy-based red hazard detection: {str(e)}")
            return False, np.zeros((1, 1), dtype=np.uint8), 0
    
    def comprehensive_safety_analysis(self, frame: np.ndarray) -> Dict:
        """
        Perform comprehensive safety analysis on the frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing comprehensive safety analysis results
        """
        try:
            results = {
                'persons_detected': [],
                'ppe_violations': [],
                'zone_intrusions': [],
                'hazards_detected': [],
                'overall_risk_level': 'low',
                'recommended_action': 'continue_operation'
            }
            
            # Detect persons
            persons = self.detect_persons(frame)
            results['persons_detected'] = persons
            
            # Analyze each detected person
            for person in persons:
                # Check PPE compliance
                ppe_status = self.detect_ppe(frame, person['bbox'])
                missing_ppe = [item for item, status in ppe_status.items() if not status]
                
                if missing_ppe:
                    results['ppe_violations'].append({
                        'person_id': len(results['ppe_violations']),
                        'missing_ppe': missing_ppe,
                        'bbox': person['bbox']
                    })
                
                # Check zone intrusions
                intrusion_status = self.check_zone_intrusion(person['center'])
                if intrusion_status['machine_zone'] or intrusion_status['restricted_areas']:
                    results['zone_intrusions'].append({
                        'person_id': len(results['zone_intrusions']),
                        'machine_zone': intrusion_status['machine_zone'],
                        'restricted_areas': intrusion_status['restricted_areas'],
                        'bbox': person['bbox']
                    })
            
            # Detect red hazards
            red_hazard, red_mask, red_pixels = self.detect_red_hazards(frame)
            if red_hazard:
                results['hazards_detected'].append({
                    'type': 'red_hazard',
                    'pixel_count': red_pixels,
                    'mask': red_mask
                })
            
            # Classify other hazards
            hazard_classification = self.classify_hazard(frame)
            if hazard_classification['confidence'] > 0.7:
                results['hazards_detected'].append({
                    'type': hazard_classification['hazard_type'],
                    'confidence': hazard_classification['confidence']
                })
            
            # Determine overall risk level
            risk_score = 0
            if results['ppe_violations']:
                risk_score += len(results['ppe_violations']) * 2
            if results['zone_intrusions']:
                risk_score += len(results['zone_intrusions']) * 3
            if results['hazards_detected']:
                risk_score += len(results['hazards_detected']) * 4
            
            if risk_score >= 8:
                results['overall_risk_level'] = 'critical'
                results['recommended_action'] = 'emergency_stop'
            elif risk_score >= 5:
                results['overall_risk_level'] = 'high'
                results['recommended_action'] = 'warning_alert'
            elif risk_score >= 2:
                results['overall_risk_level'] = 'medium'
                results['recommended_action'] = 'caution_alert'
            else:
                results['overall_risk_level'] = 'low'
                results['recommended_action'] = 'continue_operation'
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive safety analysis: {str(e)}")
            return {
                'persons_detected': [],
                'ppe_violations': [],
                'zone_intrusions': [],
                'hazards_detected': [],
                'overall_risk_level': 'unknown',
                'recommended_action': 'continue_operation'
            }
    
    def download_kaggle_dataset(self, dataset_name: str, target_dir: str = "datasets"):
        """
        Download dataset from Kaggle for training.
        
        Args:
            dataset_name: Name of the Kaggle dataset
            target_dir: Directory to save the dataset
        """
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
            logger.info(f"Dataset {dataset_name} downloaded successfully to {target_dir}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
    
    def train_custom_model(self, dataset_path: str, model_type: str = "yolo"):
        """
        Train a custom model on the provided dataset.
        
        Args:
            dataset_path: Path to the training dataset
            model_type: Type of model to train ("yolo", "faster_rcnn", etc.)
        """
        try:
            if model_type == "yolo":
                # Train YOLO model
                model = YOLO('yolov8n.pt')
                model.train(data=dataset_path, epochs=100, imgsz=640)
                logger.info("Custom YOLO model training completed")
            else:
                logger.warning(f"Model type {model_type} not supported yet")
        except Exception as e:
            logger.error(f"Error training custom model: {str(e)}") 