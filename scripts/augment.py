import pathlib
import random
from PIL import Image, ImageEnhance, ImageOps, GaussianBlur


def augment_image(in_path: pathlib.Path, out_dir: pathlib.Path):
    """Generate a single augmented version of the image at `in_path`."""
    try:
        img = Image.open(in_path)

        #Random rotation
        angle = random.choice([-15, -10, -5, 0, 5, 10, 15])
        img = img.rotate(angle, expand=True)

        #Horizontal flip (50% chance)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)  # ← FIXED: PIL uses mirror(), not fliplr()

        #Brightness & contrast tweaks
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        #Optional light Gaussian blur
        if random.random() < 0.3:
            img = img.filter(GaussianBlur(radius=random.uniform(1, 2)))

        #Save with a unique suffix
        suffix = random.randint(1, 9999)
        out_path = out_dir / f"{in_path.stem}_aug_{suffix}.png"  # ← FIXED: use in_path, not img_path
        img.save(out_path)
    except Exception as e:
        print(f"[WARN] Failed to augment {in_path}: {e}")


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
