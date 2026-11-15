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
