"""Benchmark LeakLens on street photos from US cities.

Step 1, fetch photos (Wikimedia Commons, freely licensed, credited in the manifest):
    python scripts/benchmark.py fetch

Step 2, run the pipeline on every photo and log the results:
    python scripts/benchmark.py run

Outputs go to benchmark/results/<timestamp>/:
    results.csv   one row per photo
    summary.md    accuracy, distance error, risk before/after, timing

Metadata (EXIF) is stripped before analysis by default, so the numbers
measure what the *pixels* give away. Pass --keep-exif to include it.
"""

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "benchmark"
CITIES = BENCH / "cities.csv"
IMAGES = BENCH / "images"
MANIFEST = IMAGES / "manifest.json"
API = "https://commons.wikimedia.org/w/api.php"
UA = "LeakLens-benchmark/0.1 (https://github.com/jayinfinia/leaklens)"
OK_LICENSES = ("CC0", "Public domain", "CC BY", "CC BY-SA", "PD")


# ------------------------------------------------------------ helpers
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def load_cities() -> list[dict]:
    with CITIES.open() as f:
        return [dict(r, lat=float(r["lat"]), lon=float(r["lon"])) for r in csv.DictReader(f)]


def _get(url: str, params: dict | None = None) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _strip_html(s: str) -> str:
    out, inside = [], False
    for ch in s or "":
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out.append(ch)
    return " ".join("".join(out).split())


# ------------------------------------------------------------ fetch
def search_commons(city: dict, per_city: int) -> list[dict]:
    place = city["city"] if city["state"].startswith("Washington, D") else f"{city['city']} {city['state']}"
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{place} street filetype:bitmap", "gsrlimit": 25,
        "prop": "imageinfo|coordinates", "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1280, "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl",
    }
    pages = json.loads(_get(API, params)).get("query", {}).get("pages", {})
    picks = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if info.get("mime") != "image/jpeg" or info.get("width", 0) < 800:
            continue
        if not lic.startswith(OK_LICENSES):
            continue
        coords = (page.get("coordinates") or [{}])[0]
        picks.append({
            "title": page["title"],
            "page_url": info.get("descriptionurl"),
            "image_url": info.get("thumburl") or info.get("url"),
            "license": lic,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
            "photo_lat": coords.get("lat"),
            "photo_lon": coords.get("lon"),
        })
        if len(picks) >= per_city:
            break
    return picks


def fetch(per_city: int) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for city in load_cities():
        try:
            picks = search_commons(city, per_city)
        except Exception as exc:
            print(f"  ! {city['city']}: search failed ({exc})")
            continue
        if not picks:
            print(f"  ! {city['city']}: no usable photo found")
        for i, pick in enumerate(picks):
            name = f"{slug(city['city'])}-{i + 1}.jpg"
            try:
                (IMAGES / name).write_bytes(_get(pick["image_url"]))
            except Exception as exc:
                print(f"  ! {name}: download failed ({exc})")
                continue
            manifest.append({"file": name, **city, **pick})
            print(f"  ok {name}  <- {pick['title']} ({pick['license']}, {pick['artist'][:40]})")
        time.sleep(0.5)  # be polite to the API
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} photos saved to {IMAGES}")


# ------------------------------------------------------------ run
def run(keep_exif: bool, limit: int | None) -> Path:
    from app import config
    from app.pipeline import analyze, load_image
    from app.pipeline.exif import strip_metadata
    from app.pipeline.safe import plan
    from app.pipeline.scrub import blur_regions

    if not MANIFEST.exists():
        sys.exit("No photos yet. Run:  python scripts/benchmark.py fetch")
    items = json.loads(MANIFEST.read_text())[:limit]

    out_dir = BENCH / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    rows = []

    if not config.MOCK_MODE:
        from app.models import warm_up

        print("Loading models…")
        warm_up()

    for n, item in enumerate(items, 1):
        data = (IMAGES / item["file"]).read_bytes()
        raw, upright = load_image(data)
        had_gps = bool(raw.getexif().get_ifd(0x8825)) if raw.getexif() else False
        if not keep_exif:
            raw = upright = strip_metadata(upright)

        truth_lat = item["photo_lat"] if item.get("photo_lat") is not None else item["lat"]
        truth_lon = item["photo_lon"] if item.get("photo_lon") is not None else item["lon"]

        t0 = time.perf_counter()
        a = analyze(raw, upright, explain=True)
        t_analyze = time.perf_counter() - t0

        t0 = time.perf_counter()
        ids, note = plan(upright, a)
        boxes = [c.box for c in a.clues if c.id in ids]
        clean = blur_regions(upright, boxes)
        after = analyze(clean, clean, explain=False)
        t_safe = time.perf_counter() - t0

        g = a.geolocation
        regions = [r.name for r in g.region]
        pin = g.points[0] if g.points else None
        km = haversine_km(pin.lat, pin.lon, truth_lat, truth_lon) if pin else None
        top_clue = max(a.clues, key=lambda c: c.impact or 0, default=None)

        row = {
            "file": item["file"], "city": item["city"], "state": item["state"],
            "had_gps_exif": had_gps,
            "pred_country": g.country[0].name if g.country else "",
            "country_ok": bool(g.country) and g.country[0].name == "United States",
            "pred_state": regions[0] if regions else "",
            "state_prob": g.region[0].prob if g.region else "",
            "state_top1": bool(regions) and regions[0] == item["state"],
            "state_top5": item["state"] in regions,
            "pin_lat": pin.lat if pin else "", "pin_lon": pin.lon if pin else "",
            "pin_error_km": round(km, 1) if km is not None else "",
            "n_object_clues": sum(c.kind == "object" for c in a.clues),
            "n_text_clues": sum(c.kind == "text" for c in a.clues),
            "text_found": " | ".join(c.label for c in a.clues if c.kind == "text")[:200],
            "top_clue": (f"{top_clue.category}:{top_clue.label}"[:60] if top_clue else ""),
            "top_clue_impact": (top_clue.impact if top_clue else ""),
            "scene": " ".join(s.category for s in a.scene),
            "risk_before": a.risk.score, "level_before": a.risk.level,
            "blurred": len(ids), "risk_after": after.risk.score, "level_after": after.risk.level,
            "after_summary": after.geolocation.summary,
            "secs_analyze": round(t_analyze, 2), "secs_safe": round(t_safe, 2),
            "license": item.get("license", ""), "artist": item.get("artist", ""),
            "source": item.get("page_url", ""),
        }
        rows.append(row)
        print(f"[{n:>2}/{len(items)}] {item['city']:<16} pred={row['pred_state'] or row['pred_country']:<18} "
              f"{'✓' if row['state_top1'] else '·'} err={row['pin_error_km'] or '-':>7} km  "
              f"risk {row['risk_before']:>3} → {row['risk_after']:>3}  ({t_analyze:.1f}s)")

    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.md").write_text(summarize(rows, keep_exif, config.MOCK_MODE))
    print(f"\nLogged to {out_dir}")
    return out_dir


def _pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"


def summarize(rows: list[dict], keep_exif: bool, mock: bool) -> str:
    kms = [r["pin_error_km"] for r in rows if r["pin_error_km"] != ""]
    drop = [r["risk_before"] - r["risk_after"] for r in rows]
    lines = [
        f"# LeakLens benchmark, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Photos: {len(rows)} from {len({r['city'] for r in rows})} US cities",
        f"- EXIF: {'kept' if keep_exif else 'stripped before analysis (pixels only)'}",
        f"- Mode: {'MOCK (canned results, not real accuracy)' if mock else 'real models'}",
        "",
        "## Where the models placed the photos",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Country = United States | {_pct(r['country_ok'] for r in rows)} |",
        f"| Correct state, top 1 | {_pct(r['state_top1'] for r in rows)} |",
        f"| Correct state, top 5 | {_pct(r['state_top5'] for r in rows)} |",
    ]
    if kms:
        lines += [
            f"| Map pin median error | {statistics.median(kms):.0f} km |",
            f"| Pin within 25 km (city) | {_pct(k <= 25 for k in kms)} |",
            f"| Pin within 200 km (region) | {_pct(k <= 200 for k in kms)} |",
            f"| Pin within 750 km | {_pct(k <= 750 for k in kms)} |",
        ]
    lines += [
        "",
        "## Risk and the fix",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean risk before | {statistics.mean(r['risk_before'] for r in rows):.0f} |",
        f"| Mean risk after Make it safe | {statistics.mean(r['risk_after'] for r in rows):.0f} |",
        f"| Mean drop | {statistics.mean(drop):.0f} pts |",
        f"| Low risk after fix | {_pct(r['level_after'] == 'low' for r in rows)} |",
        f"| Photos with readable text | {_pct(r['n_text_clues'] > 0 for r in rows)} |",
        f"| Mean analyze time | {statistics.mean(r['secs_analyze'] for r in rows):.1f} s |",
        f"| Mean make-safe time | {statistics.mean(r['secs_safe'] for r in rows):.1f} s |",
        "",
        "## Per photo",
        "",
        "| City | Predicted | State ✓ | Pin err (km) | Top clue | Risk before → after |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['city']} | {r['pred_state'] or r['pred_country']} | {'✓' if r['state_top1'] else ''} | "
            f"{r['pin_error_km'] or '-'} | {r['top_clue'] or '-'} | {r['risk_before']} → {r['risk_after']} |")
    lines += ["", "## Photo credits", ""]
    for r in rows:
        lines.append(f"- {r['file']}: {r['artist'] or 'unknown'}, {r['license']}, {r['source']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download freely licensed street photos")
    f.add_argument("--per-city", type=int, default=1)
    r = sub.add_parser("run", help="analyze every photo and log results")
    r.add_argument("--keep-exif", action="store_true")
    r.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.per_city)
    else:
        run(args.keep_exif, args.limit)


if __name__ == "__main__":
    main()
