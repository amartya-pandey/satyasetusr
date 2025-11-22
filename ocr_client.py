import os
import requests
import json
import time
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OCRClient:
    """Client for OCR.space REST API service."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OCR client.
        
        Args:
            api_key: OCR.space API key. If None, reads from OCRSPACE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('OCRSPACE_API_KEY')
        if not self.api_key:
            raise ValueError("OCR.space API key not provided. Set OCRSPACE_API_KEY environment variable.")
        
        self.base_url = "https://api.ocr.space/parse/image"
        self.timeout = 30
        
    def extract_text_from_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Extract text from an image file using OCR.space API.
        
        Args:
            file_path: Path to the image file
            **kwargs: Additional OCR parameters
            
        Returns:
            Dictionary containing OCR results, extracted text, and bounding boxes
        """
        try:
            with open(file_path, 'rb') as file:
                return self.extract_text_from_bytes(file.read(), **kwargs)
        except FileNotFoundError:
            return {
                'success': False,
                'error': f'File not found: {file_path}',
                'extracted_text': '',
                'bounding_boxes': []
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading file: {str(e)}',
                'extracted_text': '',
                'bounding_boxes': []
            }
    
    def extract_text_from_bytes(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """
        Extract text from image bytes using OCR.space API.
        
        Args:
            image_bytes: Raw image bytes
            **kwargs: Additional OCR parameters (language, overlay, etc.)
            
        Returns:
            Dictionary containing OCR results, extracted text, and bounding boxes
        """
        # Preprocess image if it's too large (OCR.space free tier limit is 1MB)
        processed_bytes = self._preprocess_image(image_bytes)
        
        # Default parameters for OCR
        payload = {
            'apikey': self.api_key,
            'language': kwargs.get('language', 'eng'),
            'isOverlayRequired': kwargs.get('overlay', True),  # Get bounding boxes
            'OCREngine': kwargs.get('engine', 2),  # Engine 2 is generally better
            'scale': kwargs.get('scale', True),
            'isTable': kwargs.get('table', False),
            'filetype': kwargs.get('filetype', 'auto')
        }
        
        files = {
            'file': ('certificate.jpg', processed_bytes, 'image/jpeg')
        }
        
        try:
            print("Calling OCR.space API...")
            print(f"Image size: {len(image_bytes)} bytes")
            response = requests.post(
                self.base_url,
                data=payload,
                files=files,
                timeout=self.timeout
            )
            
            print(f"API Response Status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            print(f"API Response: {result}")
            return self._process_ocr_result(result)
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'OCR request timed out',
                'extracted_text': '',
                'bounding_boxes': []
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'OCR request failed: {str(e)}',
                'extracted_text': '',
                'bounding_boxes': []
            }
        except json.JSONDecodeError:
            return {
                'success': False,
                'error': 'Invalid JSON response from OCR service',
                'extracted_text': '',
                'bounding_boxes': []
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'extracted_text': '',
                'bounding_boxes': []
            }
    
    def _process_ocr_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw OCR.space API response.
        
        Args:
            result: Raw API response
            
        Returns:
            Processed OCR result with extracted text and bounding boxes
        """
        # Check for processing errors
        if result.get('IsErroredOnProcessing', False):
            error_msg = result.get('ErrorMessage', ['Unknown OCR error'])
            if isinstance(error_msg, list):
                error_msg = '; '.join(error_msg)
            
            # Provide specific guidance for common errors
            if 'E301' in error_msg:
                error_msg += " (Try: reduce image size <1MB, use JPG/PNG format, or try a different image)"
            elif 'E302' in error_msg:
                error_msg += " (API key issue - check your OCR.space account)"
            elif 'E303' in error_msg:
                error_msg += " (Rate limit exceeded - wait a moment and try again)"
            
            return {
                'success': False,
                'error': error_msg,
                'extracted_text': '',
                'bounding_boxes': []
            }
        
        # Process successful result
        extracted_text = ""
        bounding_boxes = []
        
        parsed_results = result.get('ParsedResults', [])
        if parsed_results:
            parsed_result = parsed_results[0]
            extracted_text = parsed_result.get('ParsedText', '').strip()
            
            # Extract bounding boxes if available
            text_overlay = parsed_result.get('TextOverlay', {})
            if text_overlay and 'Lines' in text_overlay:
                for line in text_overlay['Lines']:
                    for word in line.get('Words', []):
                        bounding_boxes.append({
                            'text': word.get('WordText', ''),
                            'left': word.get('Left', 0),
                            'top': word.get('Top', 0),
                            'width': word.get('Width', 0),
                            'height': word.get('Height', 0)
                        })
        
        return {
            'success': True,
            'raw_result': result,
            'extracted_text': self._clean_text(extracted_text),
            'bounding_boxes': bounding_boxes,
            'confidence': self._calculate_confidence(result)
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the extracted text by removing extra whitespace and formatting.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but keep paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Preprocess image to ensure compatibility with OCR.space API.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Processed image bytes
        """
        try:
            from PIL import Image
            import io
            
            # Check file size (1MB = 1048576 bytes)
            max_size = 1048576  # 1MB
            
            if len(image_bytes) <= max_size:
                return image_bytes
            
            print(f"Image too large ({len(image_bytes)} bytes), resizing...")
            
            # Open image and resize if needed
            image = Image.open(io.BytesIO(image_bytes))
            
            # Calculate new dimensions to stay under size limit
            quality = 85
            while True:
                output = io.BytesIO()
                image.save(output, format='JPEG', quality=quality, optimize=True)
                output_bytes = output.getvalue()
                
                if len(output_bytes) <= max_size or quality <= 20:
                    print(f"Resized to {len(output_bytes)} bytes (quality: {quality})")
                    return output_bytes
                
                quality -= 10
                
        except Exception as e:
            print(f"Image preprocessing failed: {e}, using original")
            return image_bytes
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score from OCR result.
        
        Args:
            result: OCR API response
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            parsed_results = result.get('ParsedResults', [])
            if parsed_results and len(parsed_results) > 0:
                # Simple heuristic: longer text usually means better OCR
                text_length = len(parsed_results[0].get('ParsedText', ''))
                if text_length > 100:
                    return 0.9
                elif text_length > 50:
                    return 0.7
                elif text_length > 20:
                    return 0.5
                else:
                    return 0.3
        except:
            pass
        
        return 0.5  # Default moderate confidence

# Convenience function for quick OCR
def extract_text(image_path_or_bytes: Union[str, bytes], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to extract text from image.
    
    Args:
        image_path_or_bytes: Path to image file or raw bytes
        **kwargs: OCR parameters
        
    Returns:
        OCR result dictionary
    """
    client = OCRClient()
    
    if isinstance(image_path_or_bytes, str):
        return client.extract_text_from_file(image_path_or_bytes, **kwargs)
    else:
        return client.extract_text_from_bytes(image_path_or_bytes, **kwargs)
