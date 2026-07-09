"""
Data augmentation script for the PixProof project.

Creates 3 augmented versions of each image in the `real/` and `fake/`
folders, saving them under `augmented_real/` and `augmented_fake/`
respectively.

Usage:
    python scripts/augment.py
"""

import pathlib
import random
from PIL import Image, ImageEnhance, ImageOps, GaussianBlur


def augment_image(in_path: pathlib.Path, out_dir: pathlib.Path):
    """Generate a single augmented version of the image at `in_path`."""
    img = Image.open(in_path)

    # 1️⃣ Random rotation
    angle = random.choice([-15, -10, -5, 0, 5, 10, 15])
    img = img.rotate(angle, expand=True)

    # 2️⃣ Horizontal flip (50% chance)
    if random.random() < 0.5:
        img = img.fliplr()

    # 3️⃣ Brightness & contrast tweaks
    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 4️⃣ Optional light Gaussian blur
    if random.random() < 0.3:
        img = img.filter(GaussianBlur(radius=random.uniform(1, 2)))

    # 5️⃣ Save with a unique suffix
    suffix = random.randint(1, 9999)
    out_path = out_dir / f"{img_path.stem}_aug_{suffix}.png"
    img.save(out_path)


def main():
    # Process both classes
    for cls in ("real", "fake"):
        src_dir = pathlib.Path(cls)
        out_dir = pathlib.Path(f"augmented_{cls}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_dir.glob("*.[jp][pn]g"):   # matches .jpg and .png
            for _ in range(3):  # create 3 augmented copies per original
                augment_image(img_path, out_dir)


if __name__ == "__main__":
    main()