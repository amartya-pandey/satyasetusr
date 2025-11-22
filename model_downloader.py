"""
Download ViT model from Hugging Face Hub on first run
"""
import os
from pathlib import Path
import requests
import logging

logger = logging.getLogger(__name__)

def download_vit_model():
    """Download ViT model from Hugging Face if not present"""
    model_path = Path("vit_seal_checker.pth")
    
    if model_path.exists():
        logger.info("ViT model already exists")
        return str(model_path)
    
    try:
        # Hugging Face model URL - use resolve instead of blob for direct download
        default_url = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/vit_seal_checker.pth"
        
        # Try Streamlit secrets first, then environment variable
        try:
            import streamlit as st
            model_url = st.secrets.get("VIT_MODEL_URL", default_url)
        except:
            model_url = os.getenv("VIT_MODEL_URL", default_url)
        
        if not model_url:
            logger.warning("VIT_MODEL_URL not set in secrets - using demo mode")
            return None
        
        logger.info(f"Downloading ViT model from {model_url}")
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading {total_size / (1024**3):.2f} GB model...")
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * 1024 * 1024) == 0:  # Log every 100MB
                        logger.info(f"Downloaded {downloaded / (1024**3):.2f} GB")
        
        logger.info("ViT model downloaded successfully")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Failed to download ViT model: {e}")
        return None

if __name__ == "__main__":
    download_vit_model()
