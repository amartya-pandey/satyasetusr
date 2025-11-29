"""
FastAPI Certificate Verification API
Seamlessly integrates with any website frontend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import tempfile
import os
import logging
import time

# Import existing components
try:
    from ocr_client import OCRClient
    from verifier import CertificateVerifier
    from yolo_seal_detector import YOLOSealDetector
    from vit_seal_classifier import ViTSealClassifier
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import: {e}")
    COMPONENTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Certificate Verification API",
    description="AI-Powered Certificate Authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow any website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded once)
yolo_detector = None
vit_classifier = None
ocr_client = None
verifier = None
MODELS_LOADED = False

@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    global yolo_detector, vit_classifier, ocr_client, verifier, MODELS_LOADED
    
    if not COMPONENTS_AVAILABLE:
        logger.error("Components unavailable")
        return
    
    # Check if we should skip heavy model loading (for free tier with limited RAM)
    skip_models = os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true"
    
    try:
        logger.info("Initializing API components...")
        
        if skip_models:
            logger.warning("Skipping AI model loading (SKIP_MODEL_LOADING=true)")
            logger.info("API will run with OCR and database verification only")
            yolo_detector = None
            vit_classifier = None
        else:
            # Load YOLO detector from Hugging Face
            logger.info("Loading YOLO model from Hugging Face...")
            yolo_detector = YOLOSealDetector()
            if hasattr(yolo_detector, 'load_model'):
                yolo_detector.load_model()
            logger.info("YOLOv8 loaded and ready")
            
            # Load ViT classifier from Hugging Face
            logger.info("Loading ViT model from Hugging Face...")
            vit_classifier = ViTSealClassifier()
            if hasattr(vit_classifier, 'load_model'):
                vit_classifier.load_model()
            logger.info("ViT classifier loaded and ready")
        
        # Initialize OCR client (lightweight)
        ocr_client = OCRClient()
        logger.info("OCR client initialized")
        
        # Initialize database verifier (lightweight)
        verifier = CertificateVerifier()
        logger.info("Database verifier initialized")
        
        MODELS_LOADED = True
        logger.info("API ready for requests!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        MODELS_LOADED = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Certificate Verification API",
        "version": "1.0.0",
        "status": "online",
        "models_loaded": MODELS_LOADED,
        "endpoints": {
            "verify": "POST /api/verify",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if MODELS_LOADED else "loading",
        "models": {
            "yolo": yolo_detector is not None,
            "vit": vit_classifier is not None,
            "ocr": ocr_client is not None,
            "db": verifier is not None
        }
    }

@app.post("/api/verify")
async def verify_certificate(
    file: UploadFile = File(...),
    enable_seal_verification: bool = True
):
    """
    Verify certificate image
    
    Args:
        file: Certificate image (PNG/JPG/JPEG)
        enable_seal_verification: Enable AI seal detection
    
    Returns:
        JSON with verification results
    """
    
    if not MODELS_LOADED:
        raise HTTPException(503, "Models loading, try again")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, f"Invalid file type: {file.content_type}")
    
    file_bytes = await file.read()
    
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")
    
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file")
    
    try:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"cert_{int(time.time())}.{file.filename.split('.')[-1]}")
        
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        
        # Step 1: OCR
        logger.info("Running OCR...")
        ocr_result = ocr_client.extract_text_from_bytes(file_bytes, language='eng')
        
        if not ocr_result.get('success'):
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "OCR failed",
                    "message": ocr_result.get('error', 'Text extraction failed')
                }
            )
        
        # Step 2: Database verification
        logger.info("Verifying database...")
        verification_result = verifier.verify_certificate(ocr_result, file.filename)
        
        # Step 3: Seal detection
        seal_result = None
        if enable_seal_verification:
            logger.info("Detecting seals...")
            
            try:
                summary = yolo_detector.get_detection_summary(temp_path, confidence_threshold=0.5)
                
                if summary['total_seals'] > 0:
                    fake_count = summary['class_distribution'].get('fake', 0)
                    true_count = summary['class_distribution'].get('true', 0)
                    avg_confidence = summary['average_confidence']
                    
                    if fake_count > true_count:
                        seal_status = "Fake"
                        status = "Fail"
                        reason = f"Detected {fake_count} fake vs {true_count} authentic seals"
                    elif true_count > 0 and fake_count == 0:
                        seal_status = "Real"
                        status = "Pass"
                        reason = f"All {true_count} seals appear authentic"
                    else:
                        seal_status = "Suspicious"
                        status = "Warning"
                        reason = f"Mixed: {true_count} authentic, {fake_count} fake"
                    
                    seal_result = {
                        "status": status,
                        "seal_status": seal_status,
                        "reason": reason,
                        "confidence": avg_confidence,
                        "total_seals": summary['total_seals'],
                        "authentic_seals": true_count,
                        "fake_seals": fake_count,
                        "detection_method": "YOLOv8"
                    }
                else:
                    seal_result = {
                        "status": "Warning",
                        "seal_status": "None Detected",
                        "reason": "No seals found",
                        "confidence": 0.0,
                        "total_seals": 0
                    }
                    
            except Exception as e:
                logger.error(f"Seal error: {e}")
                seal_result = {"status": "Error", "error": str(e)}
        
        # Final decision
        ocr_decision = verification_result.get('decision', 'UNKNOWN')
        ocr_confidence = verification_result.get('final_score', 0.0)
        
        # Security first: fake seals = reject
        if seal_result and seal_result.get('seal_status') == 'Fake':
            final_decision = "FAKE"
            confidence = seal_result.get('confidence', 0.0)
            reason = "Rejected due to fake seals"
        elif ocr_decision == 'AUTHENTIC' and (not seal_result or seal_result.get('status') == 'Pass'):
            final_decision = "AUTHENTIC"
            confidence = (ocr_confidence + (seal_result.get('confidence', 0) if seal_result else 0)) / 2
            reason = "Certificate verified successfully"
        elif ocr_decision == 'SUSPICIOUS' or (seal_result and seal_result.get('status') == 'Warning'):
            final_decision = "SUSPICIOUS"
            confidence = ocr_confidence
            reason = "Requires manual review"
        else:
            final_decision = "FAKE"
            confidence = ocr_confidence
            reason = "Verification failed"
        
        # Cleanup
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return {
            "success": True,
            "decision": final_decision,
            "confidence": round(confidence, 3),
            "reason": reason,
            "details": {
                "registration_number": verification_result.get('registration_no'),
                "database_match": verification_result.get('db_record') is not None,
                "ocr_data": {
                    "decision": ocr_decision,
                    "confidence": round(ocr_confidence, 3),
                    "extracted_text": ocr_result.get('extracted_text', '')[:500],
                    "field_scores": verification_result.get('field_scores', {})
                },
                "seal_verification": seal_result,
                "extracted_fields": verification_result.get('ocr_extracted', {})
            },
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.post("/api/verify/simple")
async def verify_simple(file: UploadFile = File(...)):
    """Simplified endpoint - just decision"""
    result = await verify_certificate(file)
    
    if isinstance(result, dict) and result.get('success'):
        return {
            "decision": result['decision'],
            "confidence": result['confidence'],
            "reason": result['reason']
        }
    return result

@app.get("/api/status")
async def api_status():
    """Detailed status"""
    return {
        "api_version": "1.0.0",
        "models_loaded": MODELS_LOADED,
        "components": {
            "yolo_detector": {"loaded": yolo_detector is not None, "type": "YOLOv8"},
            "vit_classifier": {"loaded": vit_classifier is not None, "type": "ViT"},
            "ocr_client": {"loaded": ocr_client is not None, "provider": "OCR.space"},
            "database": {"loaded": verifier is not None, "type": "SQLite"}
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
