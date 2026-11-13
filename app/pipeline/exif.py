"""Layer 1: metadata. Read what the file itself gives away, and strip it."""

from PIL import ExifTags, Image

from ..schemas import ExifReport, GuessPoint

GPS_IFD = 0x8825
EXIF_IFD = 0x8769

# Fields worth showing the user (everything else is noise).
INTERESTING = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "LensModel", "BodySerialNumber", "CameraOwnerName", "Artist",
}


def _to_degrees(value) -> float:
    d, m, s = (float(x) for x in value)
    return d + m / 60.0 + s / 3600.0


def _parse_gps(gps: dict):
    try:
        lat = _to_degrees(gps[2])
        lon = _to_degrees(gps[4])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if str(gps.get(1, "N")).upper().startswith("S"):
        lat = -lat
    if str(gps.get(3, "E")).upper().startswith("W"):
        lon = -lon
    if lat == 0 and lon == 0:
        return None
    return GuessPoint(lat=round(lat, 6), lon=round(lon, 6), prob=1.0)


def read_exif(img: Image.Image) -> ExifReport:
    """Must be called on the image as opened, before exif_transpose."""
    exif = img.getexif()
    if not exif:
        return ExifReport()

    fields: dict[str, str] = {}
    merged = dict(exif.items())
    try:
        merged.update(exif.get_ifd(EXIF_IFD))
    except Exception:
        pass
    for tag, value in merged.items():
        name = ExifTags.TAGS.get(tag, str(tag))
        if name in INTERESTING:
            fields[name] = str(value).strip("\x00 ")[:120]

    gps = None
    try:
        gps_ifd = exif.get_ifd(GPS_IFD)
        if gps_ifd:
            gps = _parse_gps(gps_ifd)
    except Exception:
        gps = None

    device = " ".join(x for x in (fields.get("Make"), fields.get("Model")) if x) or None
    taken_at = fields.get("DateTimeOriginal") or fields.get("DateTime")
    return ExifReport(has_exif=True, gps=gps, taken_at=taken_at, device=device, fields=fields)


def strip_metadata(img: Image.Image) -> Image.Image:
    """Return a pixel-only copy: no EXIF, XMP, ICC or comments."""
    rgb = img.convert("RGB")
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())
