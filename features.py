from pathlib import Path
import exifread
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
import tqdm
from PIL import Image, ImageOps

# ----------------------------------------------------------------------
#   PREPROCESSING + BACKBONE
# ----------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Load a pretrained EfficientNet‑B3 (new API)
_BACKBONE = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
_BACKBONE.eval()

# ----------------------------------------------------------------------
#   SINGLE IMAGE LOADER
# ----------------------------------------------------------------------
def load_image(p: Path) -> torch.Tensor:
    """Open, resize to 224×224 and normalise a PIL image."""
    with Image.open(p) as img:
        img = img.convert("RGB")
        img = ImageOps.fit(img, (224, 224), method=Image.LANCZOS)
        # <-- ToTensor is a callable transform, so we must invoke it
        return T.ToTensor()(img)   # shape = (3, 224, 224)

# ----------------------------------------------------------------------
#    EMBEDDING EXTRACTION
# ----------------------------------------------------------------------
# Hook variable (global) that will hold the penultimate activation
_embedding_buffer = torch.empty(0)

def _penultimate_hook(module, input, output):
    global _embedding_buffer
    _embedding_buffer = output.detach()

# Register hook on the last bottleneck layer of the backbone
_PENULTIMATE_MODULE = _BACKBONE.features[-1]
_PENULTIMATE_HOOK = _PENULTIMATE_MODULE.register_forward_hook(_penultimate_hook)

def get_embedding(p: Path) -> np.ndarray:
    """Run a single image through the backbone and return the penultimate activation."""
    from PIL import Image
    tensor = load_image(p).unsqueeze(0)      # (1, 3, 224, 224)
    global _embedding_buffer
    _embedding_buffer.zero_()
    _ = _BACKBONE(tensor)
    embedding = _embedding_buffer.squeeze().cpu().numpy()
    return embedding

def get_embeddings_from_folder(folder: Path) -> tuple[np.ndarray, list[Path]]:
    """Return a matrix of embeddings (one row per image) and the list of paths."""
    embeddings = []
    paths = []
    for p in tqdm.tqdm(sorted(folder.glob("*")), desc=f"Embedding {folder.name}"):
        try:
            emb = get_embedding(p)
            embeddings.append(emb)
            paths.append(p)
        except Exception as e:                # pragma: no cover
            print(f"[WARN] Skipping {p}: {e}")
    if not embeddings:
        return np.empty((0,)), []
    return np.stack(embeddings), paths

# ----------------------------------------------------------------------
#    METADATA (EXIF / PRNU) FEATURES
# ----------------------------------------------------------------------
def extract_metadata_features(p: Path) -> np.ndarray:
    """
    Very lightweight EXIF feature – count of distinct EXIF tags.
    Returns a 1‑D NumPy array of shape (1,).
    """
    try:
        with open(p, "rb") as f:
            tags = exifread.process_file(f, details=False)
            return np.array([len(tags)], dtype=np.float32)
    except Exception as e:
        # If the file has no EXIF or is corrupted, return a zero vector
        return np.zeros((1,), dtype=np.float32)

# ----------------------------------------------------------------------
#   CLEAN‑UP HOOK
# ----------------------------------------------------------------------
def close_hooks():
    global _PENULTIMATE_HOOK
    if _PENULTIMATE_HOOK is not None:
        _PENULTIMATE_HOOK.remove()
        _PENULTIMATE_HOOK = None
