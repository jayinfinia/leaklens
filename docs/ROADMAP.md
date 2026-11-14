# LeakLens roadmap

Proposals from researching successful geolocation systems (GeoSpy/Raven, GeoVista,
Bellingcat OSM Search, DoxBench/GeoMiner, GeoShield, MegaLoc, OSV-5M).
Status: #1 and #2 are implemented; #3 to #5 are planned below.

## 1. Local AI vision model (done)
Qwen3-VL through Ollama, GeoMiner-style two-stage prompt (clue extraction, then reasoning),
structured JSON output. Reads signs OCR misses, lists objects from a fixed OSM-backed
vocabulary, proposes up to 3 places. Its readings upgrade OCR clues; its guesses become
search hints and corroboration for the OpenStreetMap lookup. Code: `app/pipeline/vlm.py`.

## 2. Bellingcat-style object proximity search (done)
Objects in the photo (from the detector and the AI model) map to OSM tags with rarity
weights. For an ambiguous sign, Overpass checks which candidate has those objects within
150 m; a clear winner (score >= 2.5 and 2x the runner-up) is confirmed.
Code: `app/pipeline/features.py`.

## 3. Verify candidates against street-level photos (planned, 1-2 days)
Goal: turn "possible match" into "verified" by checking that a candidate's streets look like
the photo.
1. Get a free Mapillary access token; add `LEAKLENS_MAPILLARY_TOKEN`.
2. For the top 3-5 candidates (text matches, AI guesses geocoded to street level), fetch up
   to 20 images per candidate with `GET graph.mapillary.com/images?bbox=...&fields=id,thumb_1024_url,computed_geometry`
   (bbox must be smaller than 0.01 degrees square, roughly 1 km; use about 300 m).
3. Embed the query photo and candidate images with MegaLoc (DINOv2 + SALAD, `gmberton/MegaLoc`);
   cosine similarity, take the best image per candidate.
4. Verified if the best candidate beats the others by a margin and clears an absolute
   threshold calibrated on the benchmark (#5). Show the matching street photo next to the upload.
5. Cache embeddings on disk; Mapillary imagery is CC-BY-SA, so credit it in the UI.
Risks: sparse Mapillary coverage in small towns (fall back silently); indoor photos.

## 4. Invisible scrub against AI geolocation (planned, research-sized)
Goal: a "hide from AI" option that keeps the photo looking unchanged, in addition to blur.
1. Reproduce GeoShield (AAAI 2026, `thinwayliu/Geoshield`): feature disentanglement,
   geo-revealing region identification, scale-adaptive perturbation.
2. Use LeakLens's own clue boxes and attribution as the region prior (replaces their
   exposure module, and is explainable).
3. Surrogates: StreetCLIP + GeoCLIP + Qwen3-VL (all local); measure transfer to a held-out
   model to avoid overfitting to our own attacker.
4. Report: geolocation error increase vs. perceptual change (LPIPS/SSIM) at budgets 4/255 and 8/255.
5. Keep blur as the guaranteed fix for readable text: perturbations do not stop a human reading a sign.
Fits the BadCLIP/adversarial line of research; candidate for a workshop paper.

## 5. Credible benchmark (planned, a few hours)
1. Add an OSV-5M test slice (Mapillary street views, strict 1 km train/test separation)
   loader to `scripts/benchmark.py`: sample 200 US images, ground truth from metadata.
2. Request DoxBench access (gated, research-use agreement) for private-setting iPhone photos.
3. Report per layer (vision models only, + OCR lookup, + AI model, + object search):
   accuracy within 1/25/200/750 km, strong-match precision (how often "strong" is right),
   possible-match rate, risk before/after Make it safe, seconds per photo.
4. Track strong-match precision above all: LeakLens should never claim a wrong exact place.
