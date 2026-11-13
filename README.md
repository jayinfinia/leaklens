# LeakLens

**See what your photo gives away before you post it.**

Stripping EXIF data isn't enough. A street sign in the background, a storefront, a license plate or the skyline out a window can pin a photo to a neighborhood, and modern vision models do it automatically. LeakLens runs that attack on your own photo first, shows you exactly which details give it away, and scrubs them.

## How it works

| Layer | What it does | Built on |
|---|---|---|
| 1. Metadata | Reads GPS, timestamp and device info from the file, then strips all of it | Pillow |
| 2. Visual clues | Draws boxes around location-revealing objects (street signs, house numbers, plates, storefronts, landmarks, transit signs) | OWLv2 (open-vocabulary detection) |
| 2. Text | Finds readable names, addresses and numbers | EasyOCR |
| 3. Geolocation | Guesses country, then US state, and drops pins on a map | StreetCLIP + GeoCLIP |
| 4. Risk + fix | Explainable 0-100 score; blurs the selected clues and re-runs everything to show the drop | |

Everything runs locally. Photos never leave the machine and nothing is stored.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (first run downloads roughly 2-3 GB of model weights)
uvicorn app.main:app --reload

# UI-only work, no model downloads: canned results
LEAKLENS_MOCK=1 uvicorn app.main:app --reload
```

Open http://localhost:8000.

Tests (always run in mock mode):

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Configuration

All settings live in `app/config.py` and can be overridden with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LEAKLENS_MOCK` | `0` | Skip models, return canned results |
| `LEAKLENS_DEVICE` | auto | `cuda`, `mps` or `cpu` |
| `LEAKLENS_DETECT_THRESHOLD` | `0.2` | OWLv2 confidence cutoff |
| `LEAKLENS_USE_GEOCLIP` | `1` | Turn off map pins if GeoCLIP is slow |
| `LEAKLENS_MAX_SIDE` | `1280` | Longest side used for analysis |

The clue categories, prompts, weights and explanations are a single dict (`CLUE_CATEGORIES`), so tuning detection means editing text rather than code.

## API

- `POST /api/analyze`: multipart `file` → metadata, clues with boxes, geolocation guesses, risk score with every contributing factor
- `POST /api/scrub`: multipart `file` + `boxes` (JSON list of `[x0, y0, x1, y1]`) → clean JPEG (base64), risk before, and a full analysis of the cleaned image
- `GET /api/health`

## Project layout

```
app/
  main.py            FastAPI routes
  config.py          settings, clue prompts and weights
  models.py          lazy, cached model loaders
  schemas.py         response models
  pipeline/
    exif.py          layer 1: read and strip metadata
    detect.py        layer 2: OWLv2 clue boxes
    ocr.py           layer 2: readable text
    geolocate.py     layer 3: StreetCLIP labels, GeoCLIP pins
    risk.py          layer 4: explainable score
    scrub.py         layer 4: pixelate + blur, strip metadata
web/index.html       single-file frontend (Leaflet map)
tests/               mock-mode tests
```

## Limitations

- Geolocation models are biased toward well-photographed, mostly Western places. A low score is not a guarantee of safety.
- The detector finds what its prompts describe. Architecture, vegetation, road markings and sun angle still leak location and aren't boxed yet.
- Blurring removes the clue, not the scene. A determined analyst may still place a photo from what's left.

## Intended use

LeakLens is a defensive tool for checking **your own** photos before sharing them. Please don't use it to locate other people.

## Credits and prior work

This project builds on the following work. Thank you to the authors.

**Models**
- **StreetCLIP**: Haas, Alberti, Skreta. *Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization* (2023). [Model](https://huggingface.co/geolocal/StreetCLIP). Licensed **CC-BY-NC-4.0 (non-commercial)**.
- **GeoCLIP**: Vivanco Cepeda, Nayak, Shah. *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization* (NeurIPS 2023). [Code](https://github.com/VicenteVivan/geo-clip).
- **OWLv2**: Minderer, Gritsenko, Houlsby. *Scaling Open-Vocabulary Object Detection* (NeurIPS 2023). [Model](https://huggingface.co/google/owlv2-base-patch16-ensemble).
- **EasyOCR**: JaidedAI. [Code](https://github.com/JaidedAI/EasyOCR).
- **CLIP**: Radford et al., OpenAI (2021).

**Related tools that inspired the UX**
- [Image Scrubber](https://github.com/everestpipkin/image-scrubber) by Everest Pipkin: browser-based EXIF removal and manual blurring for protest photos.
- [DotScramble](https://github.com/kareem2099/DotScramble): automatic plate and OCR text redaction.

**Research**
- **PIGEON**: Haas, Skreta, Alberti, Finn. *PIGEON: Predicting Image Geolocations* (CVPR 2024). [Code](https://github.com/LukasHaas/PIGEON).
- Luo et al. *Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models* (ICLR 2026). The low/medium/high risk levels are loosely adapted from its three-level visual privacy risk framework.
- Mendes et al. *Granular Privacy Control for Geolocation with Vision Language Models* (EMNLP 2024).
- *Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models* (2026). Closest prior work on the defense side. LeakLens differs by explaining each clue and letting the user decide what to remove.

## License

Code: MIT (see `LICENSE`). Model weights keep their own licenses. StreetCLIP is non-commercial, so swap it out before any commercial use.
