# LeakLens

**See what your photo gives away before you post it.**

Stripping EXIF data isn't enough. A street sign in the background, a storefront, a license plate or the skyline out a window can pin a photo to a neighborhood, and modern vision models do it automatically. LeakLens runs that attack on your own photo first, shows you exactly which details give it away, and scrubs them.

## How it works

| Layer | What it does | Built on |
|---|---|---|
| 1. Metadata | Reads GPS, timestamp and device info from the file, then strips all of it | Pillow |
| 2. Visual clues | Draws boxes around location-revealing objects (street signs, house numbers, plates, storefronts, landmarks, transit signs) | OWLv2 (open-vocabulary detection) |
| 2. Text | Reads whole signs (merges words, stitches vertical marquee letters) and weights them by what they reveal: city or street names, phone numbers and institutions count far more than "CLASS OF 2024" | EasyOCR |
| 2. Scene | Flags whole-image signals with no single spot to blur (skyline, mountains, road markings, plants, architecture) | StreetCLIP |
| 3. Geolocation | Guesses country, then US state, and drops pins on a map | StreetCLIP + GeoCLIP |
| 3b. Sign-text lookup (opt-in) | Searches detected business and place names on OpenStreetMap, with the models' region guess as a hint, exactly as an investigator would. Matches show on the map | Nominatim |
| 4. Attribution | Blurs each clue on its own and measures how much the top guess weakens, so you see which clue gave it away | StreetCLIP (batched) |
| 5. Risk + fix | Explainable 0-100 score. **Make it safe** blurs the fewest clues needed to reach low risk; or pick clues yourself. Everything is re-checked after | |

iPhone HEIC photos work too (converted for preview automatically), and the UI still works offline; only the map tiles need internet.

Everything runs locally and nothing is stored. The one exception is the opt-in sign-text lookup, which sends detected text (never the photo) to OpenStreetMap.

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

Docker:

```bash
docker build -t leaklens .
docker run -p 8000:8000 -v leaklens-models:/models leaklens
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite runs in mock mode, so CI never downloads weights. If torch and transformers are installed, `tests/test_models_offline.py` also runs the real (non-mock) code paths against tiny, randomly initialized OWLv2 and CLIP models. That checks shapes, batching, box coordinates and library compatibility (tested on transformers 4.46 and 5.x) without downloading anything.

## Benchmark

`scripts/benchmark.py` measures LeakLens on freely licensed street photos from 20 US cities (`benchmark/cities.csv`, from New York to Anchorage):

```bash
python scripts/benchmark.py fetch     # downloads 1 photo per city from Wikimedia Commons, with credits
python scripts/benchmark.py run       # analyze + Make it safe on each, logs to benchmark/results/<timestamp>/
python scripts/benchmark.py run --lookup   # also search sign text on OpenStreetMap
```

It reports country and state accuracy (top 1 and top 5), map-pin distance error (median, % within 25/200/750 km), risk before and after Make it safe, and timing. Results go to `results.csv` and `summary.md`, with photo credits. EXIF is stripped first so the numbers reflect what the pixels give away (`--keep-exif` to include it).

## Configuration

All settings live in `app/config.py` and can be overridden with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LEAKLENS_MOCK` | `0` | Skip models, return canned results |
| `LEAKLENS_DEVICE` | auto | `cuda`, `mps` or `cpu` |
| `LEAKLENS_DETECT_THRESHOLD` | `0.3` | OWLv2 confidence cutoff |
| `LEAKLENS_TEXT_LOOKUP` | `0` | Default for the OpenStreetMap sign-text lookup (the UI has a toggle) |
| `LEAKLENS_OCR_MAX_SIDE` | `2048` | OCR reads a sharper copy than the vision models |
| `LEAKLENS_USE_GEOCLIP` | `1` | Turn off map pins if GeoCLIP is slow |
| `LEAKLENS_MAX_SIDE` | `1280` | Longest side used for analysis |
| `LEAKLENS_ATTRIBUTION` | `1` | Per-clue "which clue gave it away" test |
| `LEAKLENS_ATTRIBUTION_MAX` | `8` | Max clues to attribute (one extra image pass each) |
| `LEAKLENS_SCENE_MIN_PROB` | `0.15` | Cutoff for reporting scene signals |

Clue categories and scene cues (prompts, weights and explanations) are plain dicts (`CLUE_CATEGORIES`, `SCENE_CUES`), so tuning detection means editing text rather than code.

## API

- `POST /api/analyze?explain=true`: multipart `file` → metadata, clues with boxes and impact, scene signals, geolocation guesses, risk score with every contributing factor
- `POST /api/safe`: multipart `file` → blurs the fewest clues needed to reach low risk; returns the clean JPEG (base64), which clues were blurred, risk before, a full analysis after, and a note
- `POST /api/scrub`: multipart `file` + `boxes` (JSON list of `[x0, y0, x1, y1]`) → same response, for clues the user picked
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
    scene.py         layer 2: scene signals with no box
    geolocate.py     layer 3: StreetCLIP labels (cached text, batched images), GeoCLIP pins
    attribution.py   layer 4: which clue gave it away
    risk.py          layer 5: explainable score
    safe.py          layer 5: fewest blurs to reach low risk
    scrub.py         layer 5: pixelate + blur, strip metadata
    mock.py          canned, blur-aware results for LEAKLENS_MOCK=1
web/index.html       single-file frontend (Leaflet map)
tests/               mock-mode API tests + offline real-architecture tests
scripts/check.py     run the pipeline on one photo from the terminal
```

## Limitations

- Geolocation models are biased toward well-photographed, mostly Western places. A low score is not a guarantee of safety.
- The detector finds what its prompts describe. Scene signals (architecture, vegetation, road markings) are flagged but can't be blurred away; cropping is the fix. Sun angle and shadows aren't checked.
- Attribution measures what *these* models lean on. CLIP-style models don't read, so text often shows near-zero impact even when it is the biggest giveaway to a person. That is why text is weighted by content and why the optional lookup exists.

## How the score works

Each signal gets a leak strength between 0 and 1 (roughly: the chance it alone lets someone narrow down the location). They combine like independent chances, `score = 100 * (1 - Π(1 - leak))`, so many weak duplicate boxes can't add up to certainty while one strong signal (exact GPS, a sign that matches a real address) dominates. Within a category each extra clue counts half as much as the previous one.
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
