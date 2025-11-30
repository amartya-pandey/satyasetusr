"""
Image annotation utility for certificate verification
Draws bounding boxes on certificates showing seal authenticity
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def crop_detected_seals(image_bytes, seal_detections):
    """
    Crop individual seals from certificate image.
    
    Args:
        image_bytes: Original certificate image as bytes
        seal_detections: List of seal detection results from YOLO
        
    Returns:
        List of base64 encoded cropped seal images
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        cropped_seals = []
        
        for idx, seal in enumerate(seal_detections):
            try:
                # Get bounding box coordinates
                x1, y1, x2, y2 = seal['bbox']
                
                # Add padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Crop seal region
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    # Convert to RGB
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(cropped_rgb)
                    
                    # Convert to base64
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    seal_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    cropped_seals.append({
                        'seal_number': idx + 1,
                        'class': seal['class'],
                        'confidence': seal['confidence'],
                        'image_base64': seal_base64,
                        'image_url': f"data:image/png;base64,{seal_base64}"
                    })
            except Exception as e:
                print(f"Error cropping seal {idx + 1}: {e}")
                continue
        
        return cropped_seals
        
    except Exception as e:
        print(f"Error in crop_detected_seals: {e}")
        return []


def annotate_certificate_image(image_bytes, seal_detections):
    """
    Annotate certificate image with colored bounding boxes around seals.
    
    Args:
        image_bytes: Original certificate image as bytes
        seal_detections: List of seal detection results from YOLO
        
    Returns:
        Base64 encoded annotated image
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # Convert to RGB for better color rendering
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes for each detected seal
        for seal in seal_detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = seal['bbox']
            confidence = seal['confidence']
            class_name = seal['class']
            
            # Choose color based on authenticity
            if class_name == 'true':
                color = (0, 255, 0)  # Green for authentic
                label_bg_color = (0, 180, 0)
                status = "✓ AUTHENTIC"
            elif class_name == 'fake':
                color = (255, 0, 0)  # Red for fake
                label_bg_color = (200, 0, 0)
                status = "✗ FAKE"
            else:
                color = (255, 255, 0)  # Yellow for suspicious
                label_bg_color = (200, 200, 0)
                status = "⚠ SUSPICIOUS"
            
            # Draw bounding box with thick lines
            thickness = max(2, int(min(image.shape[:2]) / 200))
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"{status} ({confidence:.1%})"
            
            # Calculate label size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(image.shape[:2]) / 1000)
            font_thickness = max(1, int(font_scale * 2))
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw label background
            label_y1 = max(y1 - label_height - baseline - 10, 0)
            label_y2 = y1
            label_x2 = min(x1 + label_width + 10, image_rgb.shape[1])
            
            cv2.rectangle(
                image_rgb,
                (x1, label_y1),
                (label_x2, label_y2),
                label_bg_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image_rgb,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
        
        # Add legend in top-right corner
        legend_height = 100
        legend_width = 200
        margin = 20
        
        legend_y1 = margin
        legend_y2 = margin + legend_height
        legend_x1 = image_rgb.shape[1] - legend_width - margin
        legend_x2 = image_rgb.shape[1] - margin
        
        # Draw semi-transparent legend background
        overlay = image_rgb.copy()
        cv2.rectangle(overlay, (legend_x1, legend_y1), (legend_x2, legend_y2), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.7, image_rgb, 0.3, 0, image_rgb)
        
        # Draw legend border
        cv2.rectangle(image_rgb, (legend_x1, legend_y1), (legend_x2, legend_y2), (100, 100, 100), 2)
        
        # Legend text
        legend_font_scale = 0.4
        legend_thickness = 1
        y_offset = legend_y1 + 20
        
        cv2.putText(image_rgb, "Seal Status:", (legend_x1 + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 0), legend_thickness)
        
        y_offset += 25
        cv2.rectangle(image_rgb, (legend_x1 + 10, y_offset - 10), 
                     (legend_x1 + 25, y_offset + 5), (0, 255, 0), -1)
        cv2.putText(image_rgb, "Authentic", (legend_x1 + 30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 0), legend_thickness)
        
        y_offset += 25
        cv2.rectangle(image_rgb, (legend_x1 + 10, y_offset - 10), 
                     (legend_x1 + 25, y_offset + 5), (255, 0, 0), -1)
        cv2.putText(image_rgb, "Fake", (legend_x1 + 30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 0), legend_thickness)
        
        # Convert annotated image to base64
        pil_image = Image.fromarray(image_rgb)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        print(f"Error annotating image: {e}")
        return None


def create_annotated_image_url(base64_image):
    """
    Create a data URL from base64 image for direct browser display.
    
    Args:
        base64_image: Base64 encoded image string
        
    Returns:
        Data URL string
    """
    return f"data:image/png;base64,{base64_image}"
