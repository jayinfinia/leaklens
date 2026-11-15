"""Offline US place names: nearest city to a coordinate, and coordinates for
a "City, State" string. No network.

Data: app/data/us_cities.csv, the 2,620 US places with population >= 20,000
from GeoNames (cities15000, CC BY 4.0, via the MIT-licensed geonamescache
package). Columns: name, state (USPS code), lat, lon, population.
"""

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "us_cities.csv"

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "Washington, D.C.",
}
_STATE_CODE = {v.lower(): k for k, v in STATES.items()} | {k.lower(): k for k in STATES} | {
    "d.c.": "DC", "dc": "DC", "district of columbia": "DC", "washington dc": "DC",
    "washington d.c.": "DC"}
_US = {"usa", "us", "u.s.", "u.s.a.", "united states", "united states of america", "america"}


@dataclass(frozen=True)
class City:
    name: str
    state: str  # USPS code
    lat: float
    lon: float
    population: int

    @property
    def label(self) -> str:
        return f"{self.name}, {self.state}"

    @property
    def state_name(self) -> str:
        return STATES.get(self.state, self.state)


def _key(name: str) -> str:
    s = name.lower().replace("saint ", "st ").replace("st. ", "st ").replace("fort ", "ft ")
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    return " ".join(s.split())


@lru_cache(maxsize=1)
def cities() -> tuple[City, ...]:
    with DATA.open(newline="") as f:
        return tuple(City(r["name"], r["state"], float(r["lat"]), float(r["lon"]), int(r["population"]))
                     for r in csv.DictReader(f))


@lru_cache(maxsize=1)
def _arrays():
    import numpy as np

    cs = cities()
    lat = np.radians([c.lat for c in cs])
    lon = np.radians([c.lon for c in cs])
    return lat, lon, np.cos(lat)


@lru_cache(maxsize=1)
def _by_name() -> dict[str, list[City]]:
    out: dict[str, list[City]] = {}
    for c in cities():  # most populous first (file order)
        keys = {_key(c.name)}
        if c.name.endswith(" City"):  # "New York City" -> "New York"
            keys.add(_key(c.name[:-5]))
        for k in keys:
            out.setdefault(k, []).append(c)
    return out


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def nearest(lat: float, lon: float) -> tuple[City, float]:
    """Closest listed city and its distance in km (vectorized, well under 1 ms)."""
    import numpy as np

    la, lo, cosla = _arrays()
    p, q = math.radians(lat), math.radians(lon)
    a = np.sin((la - p) / 2) ** 2 + math.cos(p) * cosla * np.sin((lo - q) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(1.0, a)))
    i = int(d.argmin())
    return cities()[i], float(d[i])


def state_at(lat: float, lon: float, max_km: float = 80.0) -> str:
    """USPS code of the state a point is (probably) in: that of the nearest
    listed city, or "" when nothing is close (outside the US)."""
    c, d = nearest(lat, lon)
    return c.state if d <= max_km else ""


def state_code(name: str) -> str:
    return _STATE_CODE.get(" ".join((name or "").lower().replace(",", " ").split()), "") or \
        _STATE_CODE.get((name or "").strip().lower(), "")


def find(place: str) -> City | None:
    """'Charlottesville, Virginia, USA' -> City. Tries each comma part as the
    city name, constrained by any state part. None for places outside the US
    or not in the list."""
    parts = [p.strip() for p in (place or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) > 1 and parts[-1].lower() not in _US and not state_code(parts[-1]):
        return None  # "Toronto, Ontario, Canada"
    state = ""
    for p in parts[1:]:
        code = state_code(p)
        if code:
            state = code
    if not state and parts[0].lower().startswith("washington") and "d.c" in place.lower():
        state = "DC"
    for p in parts:
        hits = _by_name().get(_key(p), [])
        if state:
            hits = [c for c in hits if c.state == state]
        if hits:
            return hits[0]
    return None


def state_center(code: str) -> City | None:
    """Population-weighted center of a state's listed cities, as a pseudo-city."""
    cs = [c for c in cities() if c.state == code]
    if not cs:
        return None
    w = sum(c.population for c in cs)
    lat = sum(c.lat * c.population for c in cs) / w
    lon = sum(c.lon * c.population for c in cs) / w
    return City(STATES.get(code, code), code, round(lat, 4), round(lon, 4), w)


def describe(lat: float, lon: float) -> tuple[str, float | None]:
    """Readable name for a point: 'Charlottesville, VA' when within 15 km of a
    listed city, 'near Charlottesville, VA' within 100 km, else ''."""
    c, d = nearest(lat, lon)
    if d <= 15:
        return c.label, round(d, 1)
    if d <= 100:
        return f"near {c.label}", round(d, 1)
    return "", None
