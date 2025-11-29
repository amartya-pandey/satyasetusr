"""
ViT Seal Classifier using ONNX Runtime
Fast inference with quantized ONNX model
"""

import numpy as np
from PIL import Image
import os
import logging

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed. Run: pip install onnxruntime")

class ViTSealClassifierONNX:
    def __init__(self, model_path=None):
        """Initialize ViT classifier with ONNX model"""
        self.model = None
        self.session = None
        
        if not ONNX_AVAILABLE:
            logging.error("ONNX Runtime not available")
            return
        
        # Default model path
        if model_path is None:
            model_path = "vit_seal_checker_quant.onnx"
        
        self.model_path = model_path
        
        # Load model
        if os.path.exists(model_path):
            self.load_model()
        else:
            logging.warning(f"Model file not found: {model_path}")
    
    def load_model(self):
        """Load ONNX model"""
        try:
            logging.info(f"Loading ONNX model from {self.model_path}...")
            
            # Create ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']  # Use CPU (add 'CUDAExecutionProvider' for GPU)
            )
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logging.info(f"✅ ONNX model loaded successfully!")
            logging.info(f"Input: {self.input_name}, Output: {self.output_name}")
            
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")
            self.session = None
    
    def preprocess_image(self, image_path):
        """Preprocess image for ViT model"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # Resize to 224x224 (ViT input size)
            image = image.resize((224, 224), Image.BILINEAR)
            
            # Convert to numpy array and normalize
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            img_array = (img_array - mean) / std
            
            # Ensure float32 dtype after normalization
            img_array = img_array.astype(np.float32)
            
            # Convert from HWC to CHW format
            img_array = img_array.transpose(2, 0, 1)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predict if seal is real or fake"""
        try:
            if self.session is None:
                return self.create_dummy_prediction("ONNX model not loaded")
            
            # Preprocess image
            input_data = self.preprocess_image(image_path)
            if input_data is None:
                return self.create_dummy_prediction("Image preprocessing failed")
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            
            # Get logits
            logits = outputs[0][0]
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get prediction (0: Fake, 1: Real)
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Format result
            seal_status = "Real" if predicted_class == 1 else "Fake"
            
            result = {
                'seal_status': seal_status,
                'confidence': confidence,
                'probabilities': {
                    'fake': float(probabilities[0]),
                    'real': float(probabilities[1])
                },
                'model_type': 'ViT-ONNX'
            }
            
            logging.info(f"✅ Prediction: {seal_status} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return self.create_dummy_prediction(str(e))
    
    def predict_multiple_seals(self, seal_paths):
        """Predict multiple seals"""
        results = []
        
        for i, seal_path in enumerate(seal_paths, 1):
            logging.info(f"Processing seal {i}/{len(seal_paths)}...")
            result = self.predict_image(seal_path)
            results.append(result)
        
        return results
    
    def create_dummy_prediction(self, error_msg=None):
        """Create dummy prediction for fallback"""
        return {
            'seal_status': 'Unknown',
            'confidence': 0.5,
            'probabilities': {'fake': 0.5, 'real': 0.5},
            'model_type': 'ViT-ONNX (unavailable)',
            'error': error_msg
        }

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = ViTSealClassifierONNX()
    
    if classifier.session:
        print("✅ ONNX model loaded successfully!")
        print(f"Model path: {classifier.model_path}")
        print(f"Input name: {classifier.input_name}")
        print(f"Output name: {classifier.output_name}")
    else:
        print("❌ Failed to load ONNX model")
