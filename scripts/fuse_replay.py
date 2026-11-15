"""Replay the fusion offline on saved benchmark evidence.

`scripts/benchmark.py run` saves one JSON per photo (results/<run>/evidence/):
the full Analysis, the fusion's candidates and GeoCLIP's verification score for
each of them. This script rebuilds the pydantic objects, runs the *current*
fuse.fuse() on them (GeoCLIP scores come from the dump, no models needed) and
reports how often the best guess lands near the truth.

    python scripts/fuse_replay.py benchmark/results/<run>/evidence
    python scripts/fuse_replay.py <dir> --set FUSE_W_CLUSTER=1.0 --text drop
    python scripts/fuse_replay.py <dir> --grid FUSE_W_CLUSTER=0.6,1.0 --grid FUSE_W_GEOCLIP=0.4,0.8

--text keep   use the sign-text matches as recorded (default)
--text drop   ignore them: measures the non-text fusion only
--text weak   treat recorded strong matches as weak ones

--grid tries every combination of the listed values (the first value of each
should be the current default). It reports the best combination on all
photos (in-sample, optimistic) and a leave-one-out estimate: each photo is
scored with the combination that does best on the other photos.
"""

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import fuse  # noqa: E402
from app.pipeline import gazetteer as gz  # noqa: E402
from app.schemas import Analysis  # noqa: E402
from scripts.benchmark import TRUTH_OVERRIDES, load_truth_overrides  # noqa: E402

CITY_KM, REGION_KM = 25.0, 200.0


# ------------------------------------------------------------ loading
def evidence_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            sub = p / "evidence"
            out += sorted((sub if sub.is_dir() else p).glob("*.json"))
        elif p.suffix == ".json":
            out.append(p)
    return out


def load_overrides(path: Path | None) -> dict[str, tuple[float, float]]:
    """Photo name (file stem) -> (lat, lon), from benchmark/truth_overrides.csv."""
    if path is None:
        return {}
    return {Path(f).stem: v for f, v in load_truth_overrides(path).items()}


class ReplayScorer:
    """Stands in for GeoScorer: returns the GeoCLIP score saved for a spot. A
    spot that was not scored in the run gets the nearest saved score within
    `tol_km`, else the lowest saved score (counted in `misses`)."""

    def __init__(self, coords: list[tuple[float, float]], scores: list[float], tol_km: float = 2.0):
        self.saved = list(zip(coords, scores))
        self.tol_km = tol_km
        self.misses = 0

    def score(self, coords):
        out = []
        floor = min((s for _, s in self.saved), default=0.0)
        for lat, lon in coords:
            d, s = min(((gz.km(lat, lon, a, b), s) for (a, b), s in self.saved), default=(math.inf, floor))
            if d > self.tol_km:
                self.misses += 1
                s = floor
            out.append(s)
        return out


class Photo:
    def __init__(self, path: Path, overrides: dict | None = None):
        ev = json.loads(path.read_text())
        self.name = path.stem
        self.truth = tuple((overrides or {}).get(self.name, ev["truth"]))
        self.analysis = Analysis.model_validate(ev["analysis"])
        f = ev.get("fuse") or {}
        cands, verify = f.get("candidates") or [], f.get("verify")
        self.scorer = (ReplayScorer([(c["lat"], c["lon"]) for c in cands], verify)
                       if cands and verify else None)
        self.verify_truth = ev.get("verify_truth")
        self.recorded = ev["analysis"].get("best")

    def matches(self, text: str):
        ms = self.analysis.text_matches
        if text == "drop":
            return []
        if text == "weak":
            return [m.model_copy(update={"confidence": "weak"}) for m in ms]
        return list(ms)

    def run(self, text: str = "keep"):
        """(best guess, candidate list) with the current config."""
        a = self.analysis
        ai = a.ai
        ms = self.matches(text)
        if ai is not None and ai.choice is not None and fuse.first_pass_confirmed(a.geolocation, ai):
            ai = ai.model_copy(update={"choice": None})  # the pipeline now skips the chooser here
        best = fuse.fuse(a.geolocation, ai, ms, self.scorer, None)
        cands = [(c["lat"], c["lon"]) for c in fuse.LAST.get("candidates", [])]
        if not cands and best is not None:  # strong text match: no candidate list was built
            cands = [(c.lat, c.lon) for c in fuse.candidates(a.geolocation, ai, ms)] + [(best.lat, best.lon)]
        return best, cands

    def err(self, lat, lon) -> float:
        return gz.km(lat, lon, *self.truth)


# ------------------------------------------------------------ evaluation
def evaluate(photos: list[Photo], text: str) -> list[dict]:
    rows = []
    for p in photos:
        best, cands = p.run(text)
        errs = [p.err(*c) for c in cands]
        rows.append({
            "photo": p.name, "err": p.err(best.lat, best.lon) if best else math.inf,
            "place": best.place if best else "-", "source": best.source if best else "-",
            "conf": best.confidence if best else 0.0, "oracle": min(errs, default=math.inf),
            "n_cands": len(cands), "lat": best.lat if best else None, "lon": best.lon if best else None,
        })
    return rows


def stats(errs: list[float]) -> dict:
    finite = [e for e in errs if math.isfinite(e)]
    return {"n": len(errs), "city": sum(e <= CITY_KM for e in errs), "region": sum(e <= REGION_KM for e in errs),
            "median": statistics.median(finite) if finite else math.inf}


def fmt_stats(label: str, s: dict) -> str:
    return (f"{label:<34} within {CITY_KM:.0f} km: {s['city']:>2}/{s['n']}   within {REGION_KM:.0f} km: "
            f"{s['region']:>2}/{s['n']}   median {s['median']:.0f} km")


def objective(errs: list[float]) -> tuple:
    """Higher is better: city hits, then region hits, then smaller log error."""
    return (sum(e <= CITY_KM for e in errs), sum(e <= REGION_KM for e in errs),
            -sum(math.log1p(min(e, 20000)) for e in errs))


def apply(settings: dict):
    for k, v in settings.items():
        if not hasattr(config, k):
            sys.exit(f"unknown config setting {k}")
        cur = getattr(config, k)
        setattr(config, k, type(cur)(v) if not isinstance(cur, bool) else str(v).lower() in ("1", "true", "yes"))


def parse_kv(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def grid_search(photos: list[Photo], text: str, grid: dict[str, list[str]]) -> tuple[list, list, list[float]]:
    """Errors for every combination, the in-sample best combo and leave-one-out errors."""
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    saved = {k: getattr(config, k) for k in keys}
    table = []  # combo index -> per-photo errors
    try:
        for c in combos:
            apply(c)
            table.append([r["err"] for r in evaluate(photos, text)])
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
    # max() keeps the first best, so ties go to the earliest combo (the defaults)
    best = max(range(len(combos)), key=lambda j: objective(table[j]))
    loo = []
    for i in range(len(photos)):
        j = max(range(len(combos)), key=lambda j: objective(table[j][:i] + table[j][i + 1:]))
        loo.append(table[j][i])
    return combos, [combos[best], table[best]], loo


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return math.nan
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def analyze(photos: list[Photo]) -> list[str]:
    """Which signals predict a right answer (within 25 km)? Uses the evidence
    as recorded, independent of the fusion weights."""
    rows, vpos, vneg = [], {}, {}
    for p in photos:
        a, g = p.analysis, p.analysis.geolocation
        ai = a.ai if a.ai is not None and not a.ai.error else None
        guess = next((x for x in (ai.guesses if ai else []) if x.lat is not None), None)
        ch = ai.choice if ai and ai.choice and not ai.choice.error and ai.choice.lat is not None else None
        regions = [gz.state_code(r.name) for r in g.region]
        truth_state = gz.state_at(*p.truth)
        total = sum(c.prob for c in g.clusters) or 1.0
        top = g.clusters[0] if g.clusters else None
        support = (sum(c.prob for c in g.clusters if gz.km(c.lat, c.lon, guess.lat, guess.lon) <= config.FUSE_AGREE_KM)
                   / total if guess else 0.0)
        if p.scorer is not None:
            cands = fuse.candidates(g, a.ai, [])
            for c, v in zip(cands, p.scorer.score([(c.lat, c.lon) for c in cands])):
                (vpos if p.err(c.lat, c.lon) <= CITY_KM else vneg).setdefault(c.source, []).append(v)
        rows.append(dict(
            ai_ok=bool(guess) and p.err(guess.lat, guess.lon) <= CITY_KM, conf=guess.confidence if guess else 0.0,
            ch=ch is not None, ch_ok=ch is not None and p.err(ch.lat, ch.lon) <= CITY_KM,
            ch_moved=ch is not None and guess is not None and gz.km(ch.lat, ch.lon, guess.lat, guess.lon) > CITY_KM,
            us=bool(regions), sure=bool(regions) and g.region[0].prob >= config.FUSE_STATE_SURE,
            t1=bool(regions) and regions[0] == truth_state, t3=truth_state in regions[:3],
            mass=top.prob / total if top else 0.0, top_ok=bool(top) and p.err(top.lat, top.lon) <= CITY_KM,
            supported=support >= 0.05))

    def line(label, sel, key):
        s = [r for r in rows if sel(r)]
        return f"| {label} | {sum(r[key] for r in s)} of {len(s)} |"

    out = ["| Signal | Right (within 25 km) |", "|---|---|",
           line("AI first guess", lambda r: True, "ai_ok"),
           line("AI chooser pick (photos where it ran)", lambda r: r["ch"], "ch_ok"),
           line("AI first guess (same photos)", lambda r: r["ch"], "ai_ok"),
           line("chooser moved away from the first guess: chooser", lambda r: r["ch_moved"], "ch_ok"),
           line("AI first guess, confidence >= 0.9", lambda r: r["conf"] >= 0.9, "ai_ok"),
           line("AI first guess, confidence < 0.9", lambda r: r["conf"] < 0.9, "ai_ok"),
           line("AI first guess with GeoCLIP pins within 40 km", lambda r: r["supported"], "ai_ok"),
           line("AI first guess without GeoCLIP pins nearby", lambda r: not r["supported"], "ai_ok"),
           line("top GeoCLIP cluster, mass >= 0.5", lambda r: r["mass"] >= 0.5, "top_ok"),
           line("top GeoCLIP cluster, mass < 0.5", lambda r: r["mass"] < 0.5, "top_ok"),
           line("StreetCLIP top state >= 50%: it is the true state", lambda r: r["sure"], "t1"),
           line("StreetCLIP top state >= 50%: true state in its top 3", lambda r: r["sure"], "t3"),
           line("StreetCLIP top state < 50% (US): true state in its top 3",
                lambda r: r["us"] and not r["sure"], "t3")]
    for src in sorted(set(vpos) | set(vneg)):
        out.append(f"| GeoCLIP verification AUC, {src} candidates (right vs wrong) | "
                   f"{_auc(vpos.get(src, []), vneg.get(src, [])):.2f} |")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="evidence folders or JSON files")
    ap.add_argument("--text", choices=("keep", "drop", "weak"), default="keep")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a config value")
    ap.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2", help="values to try")
    ap.add_argument("--truth-overrides", default=str(TRUTH_OVERRIDES), help="CSV: file, photo_lat, photo_lon")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    ap.add_argument("--analyze", action="store_true", help="also print which signals predict a right answer")
    args = ap.parse_args(argv)

    files = evidence_files(args.paths)
    if not files:
        print("no evidence files found")
        return 1
    photos = [Photo(f, load_overrides(Path(args.truth_overrides))) for f in files]
    apply(parse_kv(args.set))

    rows = evaluate(photos, args.text)
    if not args.quiet:
        print(f"{'photo':<22}{'err km':>9}  {'source':<8}{'conf':>5}  {'oracle':>8}  best guess")
        for r in rows:
            mark = "ok" if r["err"] <= CITY_KM else ("~ " if r["oracle"] <= CITY_KM else "  ")
            print(f"{r['photo']:<22}{r['err']:>9.1f}  {r['source']:<8}{r['conf']:>5.2f}  {r['oracle']:>8.1f}  "
                  f"{mark} {r['place'][:40]}")
        print("(ok: within 25 km; ~: a candidate was within 25 km but lost)")
    print(fmt_stats(f"fusion (text={args.text})", stats([r["err"] for r in rows])))
    print(fmt_stats("oracle (best candidate)", stats([r["oracle"] for r in rows])))
    same = sum(p.recorded is not None and r["lat"] is not None
               and gz.km(p.recorded["lat"], p.recorded["lon"], r["lat"], r["lon"]) < 0.5 for r, p in zip(rows, photos))
    print(f"same pick as the recorded run: {same}/{len(rows)}")
    misses = sum(p.scorer.misses for p in photos if p.scorer)
    no_verify = sum(p.scorer is None for p in photos)
    if misses or no_verify:
        print(f"note: {misses} candidate spots had no saved GeoCLIP score; {no_verify} photos had none at all")

    if args.analyze:
        print()
        print("\n".join(analyze(photos)))

    if args.grid:
        grid = {k: v.split(",") for k, v in parse_kv(args.grid).items()}
        combos, (best, best_errs), loo = grid_search(photos, args.text, grid)
        print(f"\ngrid: {len(combos)} combinations of {', '.join(grid)}")
        print(fmt_stats("default (first values)", stats(grid_search_first(photos, args.text, grid))))
        print(fmt_stats("best in-sample (optimistic)", stats(best_errs)) + f"   {best}")
        print(fmt_stats("leave-one-out", stats(loo)))
    return 0


def grid_search_first(photos, text, grid):
    saved = {k: getattr(config, k) for k in grid}
    try:
        apply({k: v[0] for k, v in grid.items()})
        return [r["err"] for r in evaluate(photos, text)]
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


if __name__ == "__main__":
    sys.exit(main())
