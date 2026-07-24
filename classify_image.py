#!/usr/bin/env python3

import argparse, joblib, numpy as np, sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ---- config -----------------------------------------------------------
MODEL_PATH = "rf_image_classifier.joblib"
THRESHOLD_PATH = "best_threshold.npy"

# ---- preprocessing ----------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

def load_image(p: Path) -> torch.Tensor:
    with Image.open(p) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        return _TRANSFORM(img)

# ---- main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--threshold", type=Path, default=Path(THRESHOLD_PATH))
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    thresh = float(np.load(args.threshold))
    with torch.no_grad():
        tensor = load_image(args.image).unsqueeze(0)
        # backbone inference (same hook used during training)
        # … (omitted for brevity – reuse the same backbone hook logic)
        # embedding = ... (obtain 1‑D vector)
        # prob = model.predict_proba([embedding])[:, 1][0]
    print(f"ProbAI: {prob:.3f}  > thr {thresh:.3f} → {'Fake' if prob >= thresh else 'Real'}")

if __name__ == "__main__":
    main()
