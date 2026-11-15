"""scripts/fuse_replay.py on a tiny synthetic evidence dump."""

import json

import pytest

from app import config
from app.pipeline import fuse
from app.schemas import (AIChoice, AIGuess, AIReading, Analysis, ExifReport, Geolocation, GuessLabel,
                         PlaceCluster, Risk)
from scripts import fuse_replay as fr

RICHMOND = (37.5538, -77.4603)
DENVER = (39.7392, -104.9847)


def _dump(path, name, truth, geo, ai, verify_at):
    """Write one evidence file like `benchmark.py run` does: the analysis plus
    the fusion's candidates and their GeoCLIP scores."""
    a = Analysis(width=1, height=1, exif=ExifReport(), clues=[], geolocation=geo, ai=ai,
                 risk=Risk(score=0, level="low", factors=[]))
    a.best = fuse.fuse(geo, ai, [], scorer=_Scorer(verify_at))
    ev = {"file": f"{name}.jpg", "city": "x", "state": "x", "truth": list(truth),
          "analysis": a.model_dump(mode="json"), "fuse": dict(fuse.LAST), "verify_truth": 0.5}
    (path / f"{name}.json").write_text(json.dumps(ev))


class _Scorer:
    def __init__(self, target):
        self.target = target

    def score(self, coords):
        from app.pipeline import gazetteer as gz

        return [round(max(0.0, 1 - gz.km(lat, lon, *self.target) / 2000), 4) for lat, lon in coords]


@pytest.fixture
def dump(tmp_path, monkeypatch):
    # Fixed weights, so the expected picks don't move when the defaults are tuned;
    # monkeypatch also undoes any --set a test passes to fr.main.
    monkeypatch.setattr(config, "FUSE_W_CLUSTER", 0.6)
    monkeypatch.setattr(config, "FUSE_W_CHOOSER", 1.2)
    us = [GuessLabel(name="United States", prob=0.9)]
    # 1) AI and GeoCLIP agree on Richmond: right
    _dump(tmp_path, "agree", RICHMOND,
          Geolocation(country=us, region=[GuessLabel(name="Virginia", prob=0.6)],
                      clusters=[PlaceCluster(lat=37.55, lon=-77.45, prob=0.5, place="Richmond, VA")]),
          AIReading(model="m", guesses=[AIGuess(place="Richmond, VA", confidence=0.9, lat=RICHMOND[0],
                                                lon=RICHMOND[1])]), RICHMOND)
    # 2) the AI says Denver, GeoCLIP's only cluster is in Richmond, the truth is Richmond
    _dump(tmp_path, "split", RICHMOND,
          Geolocation(country=us, clusters=[PlaceCluster(lat=37.55, lon=-77.45, prob=0.2, place="Richmond, VA")]),
          AIReading(model="m", guesses=[AIGuess(place="Denver, CO", confidence=0.95, lat=DENVER[0], lon=DENVER[1])],
                    choice=AIChoice(place="Denver, CO", confidence=0.9, lat=DENVER[0], lon=DENVER[1])), DENVER)
    # 3) nothing near the truth
    _dump(tmp_path, "lost", (47.6, -122.3),
          Geolocation(country=us, clusters=[PlaceCluster(lat=37.55, lon=-77.45, prob=0.2, place="Richmond, VA")]),
          None, RICHMOND)
    return tmp_path


def test_replay_reproduces_the_run_and_reports_the_oracle(dump, capsys):
    photos = [fr.Photo(f) for f in fr.evidence_files([str(dump)])]
    assert [p.name for p in photos] == ["agree", "lost", "split"]
    rows = {r["photo"]: r for r in fr.evaluate(photos, "keep")}
    assert rows["agree"]["err"] < 25 and rows["split"]["err"] > 1000 and rows["lost"]["err"] > 1000
    assert rows["split"]["oracle"] < 25 and rows["lost"]["oracle"] > 1000  # Richmond was a candidate
    assert all(p.scorer.misses == 0 for p in photos)

    assert fr.main([str(dump), "--analyze"]) == 0
    out = capsys.readouterr().out
    assert "within 25 km:  1/3" in out and "oracle" in out and "within 25 km:  2/3" in out
    assert "same pick as the recorded run: 3/3" in out
    assert "| AI first guess | 1 of 3 |" in out


def test_replay_overrides_and_truth_file(dump, tmp_path, capsys):
    over = tmp_path / "over.csv"
    over.write_text(f"file,photo_lat,photo_lon\nlost.jpg,{RICHMOND[0]},{RICHMOND[1]}\n")
    photos = [fr.Photo(f, fr.load_overrides(over)) for f in fr.evidence_files([str(dump)])]
    assert {r["photo"]: r["err"] for r in fr.evaluate(photos, "keep")}["lost"] < 25

    # a heavier cluster weight lets Richmond's GeoCLIP cluster beat the lone AI guess
    fr.main([str(dump), "--quiet", "--set", "FUSE_W_CLUSTER=10"])
    assert "within 25 km:  2/3" in capsys.readouterr().out
    assert config.FUSE_W_CLUSTER == 10.0  # --set changes the live config, as the CLI intends


def test_grid_search_reports_in_sample_and_leave_one_out(dump):
    photos = [fr.Photo(f) for f in fr.evidence_files([str(dump)])]
    before = config.FUSE_W_CLUSTER
    combos, (best, errs), loo = fr.grid_search(photos, "keep", {"FUSE_W_CLUSTER": ["0.1", "10"]})
    assert len(combos) == 2 and best == {"FUSE_W_CLUSTER": "10"}
    assert sum(e <= 25 for e in errs) == 2 and len(loo) == 3
    assert config.FUSE_W_CLUSTER == before  # restored


def test_replay_scorer_uses_saved_scores_only():
    s = fr.ReplayScorer([RICHMOND], [0.7])
    assert s.score([(37.56, -77.46), DENVER]) == [0.7, 0.7] and s.misses == 1
