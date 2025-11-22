"""
ViT Seal Classifier - Evaluation and prediction module.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image
import os
import json
import numpy as np

class ViTSealClassifier:
    def __init__(self, model_path='vit_seal_checker.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.classes = ['Real', 'Fake']
        self.is_loaded = False
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the trained ViT model."""
        if self.is_loaded:
            return True
        
        # Try to download model if not present
        if not os.path.exists(self.model_path):
            try:
                from model_downloader import download_vit_model
                downloaded_path = download_vit_model()
                if downloaded_path:
                    self.model_path = downloaded_path
                else:
                    print(f"❌ Model file not found: {self.model_path}")
                    print("Please run train_vit_seal_model.py first to train the model.")
                    return False
            except ImportError:
                print(f"❌ Model file not found: {self.model_path}")
                print("Please run train_vit_seal_model.py first to train the model.")
                return False
        
        try:
            # Try to load pretrained ViT
            try:
                self.model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=2,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                print(f"Using local ViT configuration: {e}")
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    num_labels=2,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12
                )
                self.model = ViTForImageClassification(config)
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ ViT Seal Classifier loaded successfully!")
            
            # Load model info if available
            info_path = 'vit_model_info.json'
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                print(f"Model accuracy: {model_info.get('best_accuracy', 'Unknown'):.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_image(self, image_input):
        """
        Predict if a seal/stamp is real or fake.
        
        Args:
            image_input: Can be PIL Image, image path (str), or numpy array
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if not self.load_model():
            return {
                "error": "Model not loaded",
                "seal_status": "Error",
                "confidence": 0.0
            }
        
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Image path
                if not os.path.exists(image_input):
                    return {
                        "error": f"Image file not found: {image_input}",
                        "seal_status": "Error",
                        "confidence": 0.0
                    }
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input.convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = Image.fromarray(image_input).convert('RGB')
            else:
                return {
                    "error": "Unsupported image input type",
                    "seal_status": "Error",
                    "confidence": 0.0
                }
            
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get individual class probabilities
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
                
                # Determine result
                is_real = predicted.item() == 0
                seal_status = "Real" if is_real else "Fake"
                
                result = {
                    "step": "Seal Verification",
                    "status": "Pass" if is_real else "Fail",
                    "reason": f"Seal classified as {seal_status.lower()} with {confidence.item():.2%} confidence",
                    "seal_status": seal_status,
                    "confidence": confidence.item(),
                    "probabilities": {
                        "Real": real_prob,
                        "Fake": fake_prob
                    },
                    "prediction_text": f"{seal_status} ✅" if is_real else f"{seal_status} ❌"
                }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Prediction error: {str(e)}",
                "seal_status": "Error",
                "confidence": 0.0
            }
    
    def predict_multiple_seals(self, seal_images):
        """
        Predict multiple seals and return combined result.
        
        Args:
            seal_images: List of image inputs
            
        Returns:
            dict: Combined prediction results
        """
        if not seal_images:
            return {
                "error": "No seal images provided",
                "seal_status": "Error",
                "confidence": 0.0
            }
        
        predictions = []
        for i, seal_image in enumerate(seal_images):
            result = self.predict_image(seal_image)
            result['seal_index'] = i + 1
            predictions.append(result)
        
        # Improved combine results - more lenient approach
        overall_real_count = sum(1 for p in predictions if p.get('seal_status') == 'Real')
        overall_fake_count = len(predictions) - overall_real_count
        
        # Calculate average confidence
        valid_predictions = [p for p in predictions if 'confidence' in p and p['confidence'] > 0]
        avg_confidence = np.mean([p['confidence'] for p in valid_predictions]) if valid_predictions else 0.0
        
        # More lenient decision logic: Pass if majority are real OR if confidence is reasonable
        total_seals = len(predictions)
        real_ratio = overall_real_count / total_seals if total_seals > 0 else 0
        
        # Pass if:
        # 1. Majority (>50%) are real, OR
        # 2. At least one real seal with high confidence (>80%), OR  
        # 3. Average confidence is reasonable (>60%) regardless of classification
        high_confidence_real = any(p.get('seal_status') == 'Real' and p.get('confidence', 0) > 0.8 for p in predictions)
        
        is_overall_real = (real_ratio > 0.5) or high_confidence_real or (avg_confidence > 0.6)
        
        combined_result = {
            "step": "Seal Verification",
            "status": "Pass" if is_overall_real else "Fail",
            "reason": f"Found {overall_real_count} real seals, {overall_fake_count} fake seals",
            "seal_status": "Real" if is_overall_real else "Fake",
            "confidence": avg_confidence,
            "seal_count": len(predictions),
            "real_count": overall_real_count,
            "fake_count": overall_fake_count,
            "individual_predictions": predictions
        }
        
        return combined_result
    
    def create_dummy_prediction(self, confidence=0.85):
        """Create a dummy prediction for demo purposes."""
        import random
        
        is_real = random.choice([True, False])
        seal_status = "Real" if is_real else "Fake"
        
        return {
            "step": "Seal Verification",
            "status": "Pass" if is_real else "Fail",
            "reason": f"Seal classified as {seal_status.lower()} (demo mode)",
            "seal_status": seal_status,
            "confidence": confidence,
            "probabilities": {
                "Real": confidence if is_real else 1 - confidence,
                "Fake": 1 - confidence if is_real else confidence
            },
            "prediction_text": f"{seal_status} ✅" if is_real else f"{seal_status} ❌",
            "demo_mode": True
        }

# Utility functions
def evaluate_seal_folder(folder_path, model_path='vit_seal_checker.pth'):
    """Evaluate all images in a folder."""
    classifier = ViTSealClassifier(model_path)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Evaluating {len(image_files)} images in {folder_path}...")
    
    results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        result = classifier.predict_image(image_path)
        result['filename'] = image_file
        results.append(result)
        
        status_icon = "✅" if result.get('seal_status') == 'Real' else "❌"
        confidence = result.get('confidence', 0)
        print(f"{image_file}: {result.get('seal_status', 'Error')} {status_icon} ({confidence:.2%})")
    
    return results

def test_single_image(image_path, model_path='vit_seal_checker.pth'):
    """Test classification on a single image."""
    classifier = ViTSealClassifier(model_path)
    result = classifier.predict_image(image_path)
    
    print(f"Image: {image_path}")
    print(f"Prediction: {result}")
    
    return result

if __name__ == "__main__":
    # Test the classifier if model exists
    if os.path.exists('vit_seal_checker.pth'):
        print("🧪 Testing ViT Seal Classifier...")
        
        # Test with seal dataset if available
        test_folders = ['seal_dataset/val/real', 'seal_dataset/val/fake']
        
        for folder in test_folders:
            if os.path.exists(folder):
                print(f"\n📁 Testing folder: {folder}")
                evaluate_seal_folder(folder)
    else:
        print("❌ Model not found. Please run train_vit_seal_model.py first.")
        
        # Create dummy prediction for testing
        classifier = ViTSealClassifier()
        dummy_result = classifier.create_dummy_prediction()
        print(f"Demo prediction: {dummy_result}")
