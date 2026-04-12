"""Self-contained AQI data and Exposure Credit engine.

Uses embedded realistic AQI profiles for Indian cities so the
environment works without any external API calls.  Optionally
fetches live data from Open-Meteo when network is available.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# City profiles with realistic baseline AQI values
# ---------------------------------------------------------------------------

CITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "A": {"name": "Delhi",     "lat": 28.6139, "lon": 77.2090, "baseline_aqi": 185},
    "B": {"name": "Jaipur",    "lat": 26.9124, "lon": 75.7873, "baseline_aqi": 120},
    "C": {"name": "Agra",      "lat": 27.1767, "lon": 78.0081, "baseline_aqi": 170},
    "D": {"name": "Varanasi",  "lat": 25.3176, "lon": 82.9739, "baseline_aqi": 155},
    "E": {"name": "Lucknow",   "lat": 26.8467, "lon": 80.9462, "baseline_aqi": 140},
    "F": {"name": "Kolkata",   "lat": 22.5726, "lon": 88.3639, "baseline_aqi": 95},
    "G": {"name": "Chandigarh","lat": 30.7333, "lon": 76.7794, "baseline_aqi": 75},
    "H": {"name": "Bhopal",    "lat": 23.2599, "lon": 77.4126, "baseline_aqi": 90},
    "I": {"name": "Patna",     "lat": 25.6093, "lon": 85.1376, "baseline_aqi": 160},
}

# ---------------------------------------------------------------------------
# Grade table
# ---------------------------------------------------------------------------

GRADE_TABLE = [
    {"max_aqi": 50,  "grade": "A", "label": "Pristine Air",  "credits": +10},
    {"max_aqi": 100, "grade": "B", "label": "Acceptable",    "credits": +5},
    {"max_aqi": 150, "grade": "C", "label": "Moderate Risk",  "credits": -5},
    {"max_aqi": 200, "grade": "D", "label": "High Risk",     "credits": -15},
    {"max_aqi": 300, "grade": "E", "label": "Dangerous",     "credits": -30},
    {"max_aqi": 999, "grade": "F", "label": "Hazardous",     "credits": -50},
]

DEFAULT_STARTING_CREDITS = 100
MAX_CREDITS = 1000
MIN_CREDITS = 0


# ---------------------------------------------------------------------------
# Deterministic AQI with seeded variation
# ---------------------------------------------------------------------------


def _get_aqi_for_city(code: str, seed: str = "") -> int:
    """Get AQI for a city. Uses baseline + deterministic variation from seed."""
    profile = CITY_PROFILES.get(code)
    if not profile:
        return 100
    baseline = profile["baseline_aqi"]
    # Add deterministic jitter based on seed (episode_id)
    if seed:
        h = int(hashlib.md5(f"{code}:{seed}".encode()).hexdigest()[:8], 16)
        jitter = (h % 41) - 20  # -20 to +20
    else:
        jitter = 0
    return max(10, min(400, baseline + jitter))


def get_grade_for_aqi(aqi: int) -> Dict[str, Any]:
    """Get grade info for a given AQI value."""
    for g in GRADE_TABLE:
        if aqi <= g["max_aqi"]:
            return g
    return GRADE_TABLE[-1]


def get_city_name(code: str) -> str:
    return CITY_PROFILES.get(code, {}).get("name", code)


# ---------------------------------------------------------------------------
# Edge graph
# ---------------------------------------------------------------------------

EDGE_DISTANCES: Dict[tuple, float] = {
    ("A", "B"): 280,   # Delhi - Jaipur
    ("A", "C"): 230,   # Delhi - Agra
    ("A", "G"): 250,   # Delhi - Chandigarh
    ("B", "D"): 680,   # Jaipur - Varanasi
    ("B", "H"): 520,   # Jaipur - Bhopal
    ("C", "D"): 540,   # Agra - Varanasi
    ("C", "E"): 330,   # Agra - Lucknow
    ("D", "E"): 300,   # Varanasi - Lucknow
    ("D", "F"): 680,   # Varanasi - Kolkata
    ("D", "I"): 250,   # Varanasi - Patna
    ("E", "F"): 990,   # Lucknow - Kolkata
    ("E", "H"): 500,   # Lucknow - Bhopal
    ("G", "B"): 500,   # Chandigarh - Jaipur
    ("H", "F"): 1150,  # Bhopal - Kolkata (long)
    ("I", "F"): 530,   # Patna - Kolkata
}

ADJACENCY: Dict[str, list] = {}
for (c1, c2) in EDGE_DISTANCES:
    ADJACENCY.setdefault(c1, []).append(c2)
    ADJACENCY.setdefault(c2, []).append(c1)


def get_distance(a: str, b: str) -> float:
    return EDGE_DISTANCES.get((a, b), EDGE_DISTANCES.get((b, a), 500))


@dataclass
class SegmentResult:
    from_aqi: int
    to_aqi: int
    avg_aqi: int
    grade: str
    credit_delta: int


def grade_segment(from_code: str, to_code: str, seed: str = "") -> SegmentResult:
    """Grade a route segment between two cities."""
    from_aqi = _get_aqi_for_city(from_code, seed)
    to_aqi = _get_aqi_for_city(to_code, seed)
    avg_aqi = max(from_aqi, to_aqi)  # Conservative: use worst
    g = get_grade_for_aqi(avg_aqi)
    return SegmentResult(
        from_aqi=from_aqi,
        to_aqi=to_aqi,
        avg_aqi=avg_aqi,
        grade=g["grade"],
        credit_delta=g["credits"],
    )
