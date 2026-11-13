"""Layer 4b: the fix. Blur chosen regions and drop all metadata."""

from PIL import Image, ImageFilter

from .exif import strip_metadata


def blur_regions(img: Image.Image, boxes: list[list[float]], pad: float = 0.08) -> Image.Image:
    out = img.convert("RGB").copy()
    w, h = out.size
    for box in boxes:
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        region = (
            int(max(0, x0 - px)), int(max(0, y0 - py)),
            int(min(w, x1 + px)), int(min(h, y1 + py)),
        )
        if region[2] - region[0] < 2 or region[3] - region[1] < 2:
            continue
        crop = out.crop(region)
        # Pixelate first, then blur: plain blur on text can sometimes be reversed.
        small = crop.resize((max(1, crop.width // 12), max(1, crop.height // 12)), Image.BILINEAR)
        crop = small.resize(crop.size, Image.NEAREST)
        radius = max(6, min(crop.size) // 6)
        out.paste(crop.filter(ImageFilter.GaussianBlur(radius)), region[:2])
    return strip_metadata(out)
