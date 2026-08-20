from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import torch
import os
import tempfile
import logging
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from features module
from features import load_image, get_embedding, extract_metadata_features

# ---- Configuration ----
MODEL_PATH = Path(os.getenv("MODEL_PATH", "rf_image_classifier.joblib"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", "best_threshold.npy"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB default
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- FastAPI App ----
app = FastAPI(
    title="GLITCHROT API",
    description="Image forensics engine for detecting synthetic images",
    version="1.0.0"
)

# ---- CORS Configuration ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model Loading (Lazy) ----
calibrated = None
threshold = None

@app.on_event("startup")
async def load_models():
    global calibrated, threshold
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        calibrated = joblib.load(MODEL_PATH)
        threshold = float(np.load(THRESHOLD_PATH))
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def validate_file(upload: UploadFile) -> None:
    """Validate uploaded file before processing."""
    # Check file extension
    file_ext = Path(upload.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (read without loading into memory)
    if upload.size and upload.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
        )


def predict_image(upload: UploadFile) -> dict:
    """
    Process uploaded image and return classification.
    
    Returns:
        dict: {"classification": "Fake" or "Real", "probability": float}
    """
    try:
        # Validate file
        validate_file(upload)
        
        # Read file into memory
        img_bytes = upload.file.read()
        
        if len(img_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )
        
        # Verify it's a valid image
        try:
            Image.open(__import__('io').BytesIO(img_bytes)).verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Create temporary file (thread-safe)
        with tempfile.NamedTemporaryFile(suffix=Path(upload.filename).suffix, delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            # Extract features
            emb = get_embedding(tmp_path)
            meta = extract_metadata_features(tmp_path)
            feat = np.concatenate([emb, meta]).reshape(1, -1)
            
            # Get probability
            prob = float(calibrated.predict_proba(feat)[0, 1])
            
            # Classify
            classification = "Fake" if prob >= threshold else "Real"
            
            logger.info(f"Classification: {classification} ({prob:.3f})")
            
            return {
                "classification": classification,
                "probability": prob,
                "threshold": float(threshold)
            }
        
        finally:
            # Cleanup temporary file
            if tmp_path.exists():
                tmp_path.unlink()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "GLITCHROT API"}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze uploaded image for synthetic/manipulated content.
    
    Args:
        file: Image file (JPEG, PNG, WebP, BMP)
    
    Returns:
        dict: Classification result with probability
    """
    result = predict_image(file)
    return JSONResponse(result)
