#!/usr/bin/env python3

import argparse, joblib, numpy as np, sys, torch
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

# Load backbone once
_BACKBONE = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
_BACKBONE.eval()

# Hook for embedding extraction
_embedding_buffer = torch.empty(0)

def _penultimate_hook(module, input, output):
    global _embedding_buffer
    _embedding_buffer = output.detach()

_PENULTIMATE_MODULE = _BACKBONE.features[-1]
_PENULTIMATE_HOOK = _PENULTIMATE_MODULE.register_forward_hook(_penultimate_hook)

def load_image(p: Path) -> torch.Tensor:
    with Image.open(p) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        return _TRANSFORM(img)

def get_embedding(image_path: Path) -> np.ndarray:
    """Extract embedding from image using EfficientNet backbone."""
    tensor = load_image(image_path).unsqueeze(0)
    global _embedding_buffer
    _embedding_buffer.zero_()
    with torch.no_grad():
        _ = _BACKBONE(tensor)
    return _embedding_buffer.squeeze().cpu().numpy()

# ---- main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--threshold", type=Path, default=Path(THRESHOLD_PATH))
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    model = joblib.load(args.model)
    thresh = float(np.load(args.threshold))
    
    # Extract embedding from image
    embedding = get_embedding(args.image)
    
    # Prepare feature vector (embedding only, no metadata for CLI)
    feat = embedding.reshape(1, -1)
    
    # Get probability
    prob = model.predict_proba(feat)[0, 1]
    
    # Classify
    classification = 'Fake' if prob >= thresh else 'Real'
    print(f"ProbAI: {prob:.3f}  > thr {thresh:.3f} → {classification}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup hook
        if _PENULTIMATE_HOOK is not None:
            _PENULTIMATE_HOOK.remove()
