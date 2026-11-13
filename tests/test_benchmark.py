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
