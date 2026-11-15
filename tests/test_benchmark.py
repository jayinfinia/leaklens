from scripts.benchmark import haversine_km, load_cities, summarize


def test_haversine_known_distance():
    assert abs(haversine_km(40.7128, -74.0060, 34.0522, -118.2437) - 3936) < 10  # NYC to LA


def test_twenty_cities_with_valid_states():
    from app.pipeline.geolocate import US_STATES

    cities = load_cities()
    assert len(cities) == 20
    assert all(c["state"] in US_STATES for c in cities)


def test_summary_renders():
    row = dict(file="a.jpg", city="Boston", state="Massachusetts", country_ok=True, state_top1=True,
               state_top5=True, pin_error_km=12.0, pred_state="Massachusetts", pred_country="United States",
               top_clue="street_sign:x", risk_before=70, risk_after=20, level_after="low", n_text_clues=1,
               secs_analyze=1.0, secs_safe=2.0, artist="A", license="CC BY-SA 4.0", source="url")
    md = summarize([row], keep_exif=False, mock=True)
    assert "Correct state, top 1 | 100%" in md and "Boston" in md


def test_best_guess_prefers_the_fused_answer(photo_no_gps):
    from app.pipeline import analyze, load_image
    from scripts.benchmark import best_guess

    raw, upright = load_image(photo_no_gps)
    a = analyze(raw, upright, lookup=False, use_ai=True)
    assert best_guess(a) == (a.best.lat, a.best.lon, a.best.source)
    old = a.model_copy(update={"best": None})  # analyses without fusion still work
    assert best_guess(old)[2] in ("ai", "geoclip")


def test_summary_counts_best_sources():
    base = dict(file="a.jpg", city="Boston", state="Massachusetts", country_ok=True, state_top1=True,
                state_top5=True, pin_error_km=12.0, pred_state="Massachusetts", pred_country="United States",
                top_clue="", risk_before=70, risk_after=20, level_after="low", n_text_clues=1,
                secs_analyze=1.0, secs_safe=2.0, artist="A", license="CC BY-SA 4.0", source="url")
    rows = [dict(base, best_source="chooser", best_error_km=3.0, best_place="Boston, MA", ai_choice_error_km=3.0),
            dict(base, best_source="geoclip", best_error_km=300.0)]
    md = summarize(rows, keep_exif=False, mock=True)
    assert "chooser 1, geoclip 1" in md and "AI chooser pick within 25 km | 1 of 2" in md
    assert "(chooser: Boston, MA)" in md


def test_zero_km_is_shown_as_a_distance_not_a_dash():
    from scripts.benchmark import _km

    assert _km(0.0) == "0.0" and _km(12.5) == "12.5" and _km("") == "-" and _km(None) == "-"
    base = dict(file="a.jpg", city="Charlottesville", state="Virginia", country_ok=True, state_top1=True,
                state_top5=True, pin_error_km=0.0, pred_state="Virginia", pred_country="United States",
                top_clue="", risk_before=70, risk_after=20, level_after="low", n_text_clues=1,
                secs_analyze=1.0, secs_safe=2.0, artist="A", license="CC BY-SA 4.0", source="url",
                best_source="chooser", best_error_km=0.0, best_place="Charlottesville, VA")
    md = summarize([base], keep_exif=False, mock=True)
    assert "| 0.0 (chooser: Charlottesville, VA) |" in md and "| 0.0 | - |" in md


def test_truth_overrides(tmp_path):
    from scripts.benchmark import TRUTH_OVERRIDES, apply_truth_overrides, load_truth_overrides

    shipped = load_truth_overrides(TRUTH_OVERRIDES)
    assert shipped["washington-1.jpg"] == (38.9531, -77.4565)  # Dulles Airport, not central D.C.

    f = tmp_path / "o.csv"
    f.write_text("file,photo_lat,photo_lon,note\nb.jpg,1.5,2.5,x\n,9,9,blank file\n")
    assert load_truth_overrides(f) == {"b.jpg": (1.5, 2.5)}
    assert load_truth_overrides(tmp_path / "missing.csv") == {}
    items = [{"file": "a.jpg", "lat": 1.0, "lon": 1.0, "photo_lat": None},
             {"file": "b.jpg", "lat": 3.0, "lon": 3.0, "photo_lat": 3.1, "photo_lon": 3.1}]
    out = apply_truth_overrides(items, load_truth_overrides(f))
    assert out[0] == items[0] and (out[1]["photo_lat"], out[1]["photo_lon"]) == (1.5, 2.5)
    assert items[1]["photo_lat"] == 3.1  # the manifest entries are not modified
