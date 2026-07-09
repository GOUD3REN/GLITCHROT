from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import joblib, numpy as np, torch, numpy as np
from pathlib import Path
from features import load_image, get_embedding, extract_metadata_features

app = FastAPI()

# Load model once
MODEL_PATH = Path("/home/goud3ren/rf_image_classifier.joblib")
THRESHOLD_PATH = Path("/home/goud3ren/best_threshold.npy")
calibrated = joblib.load(MODEL_PATH)
threshold = float(np.load(THRESHOLD_PATH))


def predict_image(upload: UploadFile) -> int:
    try:
        img_bytes = upload.file.read()
        img_path = Path("/tmp/uploaded.jpg")
        img_path.write_bytes(img_bytes)

        # embedding (same method used during training)
        emb = get_embedding(img_path)          # 1‑D np vector
        meta = extract_metadata_features(img_path)
        feat = np.concatenate([emb, meta])     # shape (features,)
        feat = feat.reshape(1, -1)

        prob = calibrated.predict_proba(feat)[0, 1]
        return int(prob >= threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    label = predict_image(file)
    return JSONResponse({"label": "Fake" if label else "Real"})