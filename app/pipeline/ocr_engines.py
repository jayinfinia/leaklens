"""OCR backends behind one small interface.

    readtext(PIL image)      -> [(points, text, confidence), ...]
    recognize_row(PIL image) -> (text, confidence) for a single line of text

Apple Vision (macOS) is the default when available: it is the engine
behind Live Text, runs locally, and reads stylized signage far better than
EasyOCR. EasyOCR is the cross-platform fallback.

LEAKLENS_OCR_ENGINE = auto | apple | easyocr
"""

import io
import sys
from functools import lru_cache

from .. import config


class EasyOCREngine:
    name = "easyocr"
    min_conf = 0.4

    def __init__(self):
        from ..models import ocr

        self.reader = ocr()

    def readtext(self, image):
        import numpy as np

        return self.reader.readtext(np.array(image.convert("RGB")))

    def recognize_row(self, image):
        import numpy as np

        res = self.reader.recognize(np.array(image.convert("L")), detail=1)
        if not res:
            return None
        _, text, conf = res[0]
        return text, float(conf)


class AppleVisionEngine:
    """macOS Vision framework via pyobjc (pip install pyobjc-framework-Vision)."""

    name = "apple"
    min_conf = 0.3  # Vision reports coarse confidences (0.3 / 0.5 / 1.0)

    def __init__(self):
        import Vision  # noqa: F401  (fails fast off macOS)

        self.Vision = Vision
        from Foundation import NSData

        self.NSData = NSData

    def readtext(self, image):
        V = self.Vision
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        raw = buf.getvalue()
        data = self.NSData.dataWithBytes_length_(raw, len(raw))
        handler = V.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        request = V.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(V.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        request.setRecognitionLanguages_(config.OCR_APPLE_LANGS)
        ok = handler.performRequests_error_([request], None)
        if isinstance(ok, tuple):
            ok = ok[0]
        if not ok:
            return []
        w, h = image.size
        out = []
        for obs in request.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            cand = cands[0]
            bb = obs.boundingBox()  # normalized, origin bottom-left
            x0 = bb.origin.x * w
            y0 = (1.0 - bb.origin.y - bb.size.height) * h
            x1 = x0 + bb.size.width * w
            y1 = y0 + bb.size.height * h
            out.append(([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], str(cand.string()), float(cand.confidence())))
        return out

    def recognize_row(self, image):
        res = self.readtext(image.convert("RGB"))
        if not res:
            return None
        res.sort(key=lambda r: r[0][0][0])
        return " ".join(t for _, t, _ in res), min(c for _, _, c in res)


@lru_cache(maxsize=1)
def get_engine():
    choice = config.OCR_ENGINE
    if choice in ("auto", "apple") and sys.platform == "darwin":
        try:
            return AppleVisionEngine()
        except Exception as exc:
            if choice == "apple":
                raise
            print(f"[leaklens] Apple Vision OCR unavailable ({exc}); using EasyOCR")
    return EasyOCREngine()
