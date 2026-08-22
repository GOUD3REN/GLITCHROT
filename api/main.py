from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- FastAPI App ----
app = FastAPI(title="GLITCHROT API", version="1.0.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models (DUMMY) ----
threshold = 0.5

@app.on_event("startup")
async def startup():
    logger.info("GLITCHROT API started (DUMMY MODE)")


def validate_file(upload: UploadFile):
    file_ext = Path(upload.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    if upload.size and upload.size > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")


@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {"status": "ok", "service": "GLITCHROT API"}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)):
    logger.info(f"Analyzing file: {file.filename}")
    
    try:
        validate_file(file)
        img_bytes = file.file.read()
        
        # DUMMY: Return test data
        prob = 0.65
        classification = "Fake" if prob >= threshold else "Real"
        
        logger.info(f"Result: {classification} ({prob:.3f})")
        
        return {
            "classification": classification,
            "probability": prob,
            "threshold": float(threshold)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, "Internal server error")
