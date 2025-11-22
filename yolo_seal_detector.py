"""
YOLOv8 Seal Detector - Streamlit Integration
High-performance seal detection for certificate verification
"""

import torch
import cv2
import numpy as np
import os
import time
from PIL import Image
import streamlit as st

# Handle imports with fallback for local development
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not available. Install with: pip install ultralytics")

class YOLOSealDetector:
    """
    Advanced YOLOv8-based seal detector for certificate verification.
    Integrates seamlessly with Streamlit app.
    """
    
    def __init__(self, model_path='yolo_seal_model/best.pt', device=None):
        """
        Initialize YOLOv8 seal detector.
        
        Args:
            model_path: Path to trained YOLOv8 model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_loaded = False
        self.class_names = ['fake', 'true']  # Model classes
        
        # Streamlit session state for caching
        if 'yolo_model_loaded' not in st.session_state:
            st.session_state.yolo_model_loaded = False
    
    def load_model(self):
        """Load the trained YOLOv8 model with Streamlit caching."""
        if not YOLO_AVAILABLE:
            st.error("❌ YOLOv8 not available. Please install: pip install ultralytics")
            return False
        
        # Check if model is already loaded AND model object exists
        if self.is_loaded and self.model is not None:
            return True
        
        # Check session state and reload if needed
        if st.session_state.yolo_model_loaded and self.model is None:
            # Model was loaded before but object is lost, reload it
            if os.path.exists(self.model_path):
                try:
                    self.model = YOLO(self.model_path)
                    self.is_loaded = True
                    return True
                except Exception as e:
                    st.error(f"❌ Error reloading YOLO model: {e}")
                    st.session_state.yolo_model_loaded = False
                    self.is_loaded = False
        
        # Download model from Hugging Face if not present locally
        if not os.path.exists(self.model_path):
            try:
                import requests
                from pathlib import Path
                
                # Get model URL from environment or use default
                default_url = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/best.pt"
                
                # Try Streamlit secrets first, then environment variable  
                try:
                    model_url = st.secrets.get("YOLO_MODEL_URL", default_url)
                except:
                    model_url = os.getenv("YOLO_MODEL_URL", default_url)
                
                st.info(f"📥 Downloading YOLOv8 model from Hugging Face...")
                st.write("This is a one-time download (6 MB). Future runs will use the cached model.")
                
                # Create directory if needed
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Download with progress
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(self.model_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                
                st.success(f"✅ Model downloaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to download YOLO model: {e}")
                st.info("💡 Falling back to YOLOv8 default model...")
                # Use default YOLOv8 model as fallback
                try:
                    self.model = YOLO('yolov8n.pt')  # Nano model
                    self.is_loaded = True
                    st.session_state.yolo_model_loaded = True
                    st.warning("⚠️ Using YOLOv8 default model (not trained on seals)")
                    return True
                except:
                    return False
        
        try:
            with st.spinner("🔄 Loading YOLOv8 seal detection model..."):
                self.model = YOLO(self.model_path)
                self.is_loaded = True
                st.session_state.yolo_model_loaded = True
                
            st.success(f"✅ YOLOv8 custom model loaded successfully! (Classes: {self.class_names})")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading YOLOv8 model: {e}")
            return False
    
    def detect_circular_seals(self, image_path, confidence_threshold=0.5):
        """
        Detect seals using YOLOv8 model with Streamlit integration.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detected seal regions compatible with existing system
        """
        if not self.load_model():
            return []
        
        try:
            # Validate image path
            if not os.path.exists(image_path):
                st.error(f"❌ Image file not found: {image_path}")
                return []
            
            # Run YOLOv8 inference
            with st.spinner("🔍 Detecting seals with AI..."):
                results = self.model(image_path, conf=confidence_threshold, verbose=False)
            
            # Check if results is valid
            if results is None:
                st.warning("⚠️ No detection results returned from model")
                return []
            
            detected_seals = []
            
            for r in results:
                if r is None:
                    continue
                    
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            # Extract box coordinates and info
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Ensure class_id is valid
                            if class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Calculate center and radius (for compatibility)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            radius = max(width, height) / 2
                            
                            seal_info = {
                                'center': (int(center_x), int(center_y)),
                                'radius': int(radius),
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'class': class_name,
                                'class_id': class_id,
                                'area': int(width * height),
                                'method': 'YOLOv8'
                            }
                            
                            detected_seals.append(seal_info)
                        except Exception as box_error:
                            st.warning(f"⚠️ Error processing detection box: {box_error}")
                            continue
            
            st.success(f"🎯 YOLOv8 detected {len(detected_seals)} seals with confidence > {confidence_threshold}")
            return detected_seals
            
        except Exception as e:
            st.error(f"❌ Error in YOLOv8 seal detection: {e}")
            return []
    
    def crop_seals_from_image(self, image_path, output_dir="cropped_seals", confidence_threshold=0.5):
        """
        Detect and crop seals from image with Streamlit progress tracking.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped seals
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of cropped seal file paths
        """
        detected_seals = self.detect_circular_seals(image_path, confidence_threshold)
        
        if not detected_seals:
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            st.error("❌ Could not load image for cropping")
            return []
        
        cropped_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Progress bar for cropping
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, seal in enumerate(detected_seals):
            try:
                # Update progress
                progress = (i + 1) / len(detected_seals)
                progress_bar.progress(progress)
                status_text.text(f"Cropping seal {i+1}/{len(detected_seals)}...")
                
                # Get bounding box
                x1, y1, x2, y2 = seal['bbox']
                
                # Add padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(original_image.shape[1], x2 + padding)
                y2 = min(original_image.shape[0], y2 + padding)
                
                # Crop seal region
                cropped_seal = original_image[y1:y2, x1:x2]
                
                if cropped_seal.size > 0:
                    # Generate unique filename
                    timestamp = int(time.time() * 1000) % 1000000
                    output_path = os.path.join(output_dir, f"temp_cert_{timestamp}_seal_{i+1}.png")
                    
                    # Save cropped seal
                    cv2.imwrite(output_path, cropped_seal)
                    cropped_paths.append(output_path)
                    
            except Exception as e:
                st.warning(f"⚠️ Error cropping seal {i+1}: {e}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if cropped_paths:
            st.success(f"✅ Successfully cropped {len(cropped_paths)} seals")
        
        return cropped_paths
    
    def get_detection_summary(self, image_path, confidence_threshold=0.5):
        """
        Get detailed detection summary with Streamlit visualization.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with detection summary
        """
        detected_seals = self.detect_circular_seals(image_path, confidence_threshold)
        
        # Count by class
        class_counts = {}
        for seal in detected_seals:
            class_name = seal['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(seal['confidence'] for seal in detected_seals) / len(detected_seals) if detected_seals else 0
        
        summary = {
            'total_seals': len(detected_seals),
            'class_distribution': class_counts,
            'average_confidence': avg_confidence,
            'high_confidence_seals': sum(1 for seal in detected_seals if seal['confidence'] > 0.8),
            'detection_method': 'YOLOv8',
            'model_classes': self.class_names,
            'detections': detected_seals
        }
        
        # Display summary in Streamlit
        if detected_seals:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Seals", summary['total_seals'])
            
            with col2:
                fake_count = class_counts.get('fake', 0)
                st.metric("Fake Seals", fake_count, delta=f"-{fake_count}" if fake_count > 0 else None)
            
            with col3:
                true_count = class_counts.get('true', 0)
                st.metric("Authentic Seals", true_count, delta=f"+{true_count}" if true_count > 0 else None)
            
            # Confidence metrics
            st.write("**Detection Quality:**")
            st.write(f"• Average confidence: {avg_confidence:.1%}")
            st.write(f"• High-confidence detections: {summary['high_confidence_seals']}/{len(detected_seals)}")
        else:
            st.warning("⚠️ No seals detected - certificates usually contain official seals/stamps")
        
        return summary
    
    def visualize_detections(self, image_path, confidence_threshold=0.5):
        """
        Visualize detections on the image for Streamlit display.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            PIL Image with detection boxes drawn
        """
        if not self.load_model():
            return None
        
        try:
            # Load image
            if not os.path.exists(image_path):
                st.error(f"❌ Image file not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                st.error(f"❌ Could not load image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get detections
            results = self.model(image_path, conf=confidence_threshold, verbose=False)
            
            # Check if results is valid
            if results is None:
                st.warning("⚠️ No detection results returned")
                return Image.fromarray(image_rgb)
            
            # Draw detections
            for r in results:
                if r is None:
                    continue
                    
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Ensure class_id is valid
                            if class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Choose color based on class
                            color = (0, 255, 0) if class_name == 'true' else (255, 0, 0)  # Green for true, red for fake
                            
                            # Draw bounding box
                            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(image_rgb, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(image_rgb, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        except Exception as box_error:
                            st.warning(f"⚠️ Error processing detection box: {box_error}")
                            continue
            
            return Image.fromarray(image_rgb)
            
        except Exception as e:
            st.error(f"❌ Error visualizing detections: {e}")
            # Return original image without annotations as fallback
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(image_rgb)
            except:
                pass
            return None

# Factory function for compatibility
def create_yolo_seal_detector():
    """Factory function to create YOLOv8 seal detector."""
    return YOLOSealDetector()

# Streamlit integration check
def check_yolo_integration():
    """Check if YOLOv8 integration is ready."""
    if not YOLO_AVAILABLE:
        st.warning("⚠️ YOLOv8 not installed. Install with: `pip install ultralytics`")
        return False
    
    model_path = 'yolo_seal_model/best.pt'
    
    # Check if model exists locally or can be downloaded from Hugging Face
    if not os.path.exists(model_path):
        # Check if we have the download URL configured
        try:
            model_url = st.secrets.get("YOLO_MODEL_URL", None)
        except:
            model_url = os.getenv("YOLO_MODEL_URL", None)
        
        if model_url:
            st.info("📥 YOLOv8 model will be downloaded from Hugging Face on first use (6 MB)")
            return True
        else:
            st.warning(f"⚠️ YOLOv8 model not found at: {model_path}")
            st.info("💡 **Option 1**: Model will auto-download from Hugging Face on first use")
            st.info("💡 **Option 2**: Download from Kaggle and place in yolo_seal_model/ directory")
            return True  # Still return True - will download on first use
    
    return True