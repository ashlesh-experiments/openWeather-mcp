"""
Weather MCP Server
==================
Uses OpenWeatherMap API for live data, ipinfo.io for auto location,
and ChromaDB as a persistent vector store for history + analytics.

Tools exposed:
  - get_current_weather()       → current conditions, auto-detected location
  - get_forecast(days)          → 5-day forecast with rain probability
  - get_weather_analytics()     → stats + trends from ChromaDB history
  - search_weather_history(q)   → semantic vector search over past records
  - get_weather_prediction()    → linear trend + pattern prediction from history
"""

import os
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
# PersistentClient writes to disk so data survives across sessions.
# DefaultEmbeddingFunction uses a small ONNX sentence-transformer model
# (downloads ~30 MB on first run) to embed weather description text so we
# can do semantic similarity search ("find stormy days", "hot and humid").
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_DB_PATH = os.path.expanduser("~/.weather_mcp/db")
os.makedirs(_DB_PATH, exist_ok=True)
_chroma = chromadb.PersistentClient(path=_DB_PATH)
_ef = DefaultEmbeddingFunction()
_col = _chroma.get_or_create_collection("weather_history", embedding_function=_ef)

# ── Config ────────────────────────────────────────────────────────────────────
# Get your free key at https://openweathermap.org/api
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
OWM_BASE = "https://api.openweathermap.org/data/2.5"

mcp = FastMCP("weather")


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _get_location() -> dict[str, Any]:
    """
    Auto-detect current location from IP using ipinfo.io (no key needed).
    Returns: {lat, lon, city, region, country}
    """
    async with httpx.AsyncClient() as c:
        r = await c.get("https://ipinfo.io/json", timeout=10)
        d = r.json()
    lat, lon = d.get("loc", "0,0").split(",")
    return {
        "lat": float(lat),
        "lon": float(lon),
        "city": d.get("city", "Unknown"),
        "region": d.get("region", ""),
        "country": d.get("country", "?"),
    }


async def _owm(endpoint: str, params: dict) -> dict | None:
    """
    Hit an OWM 2.5 endpoint with metric units.
    Returns parsed JSON or None on any error.
    """
    if not OPENWEATHER_API_KEY:
        return None
    full_params = {**params, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{OWM_BASE}/{endpoint}", params=full_params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


def _store_record(
    loc: dict,
    temp: float,
    feels_like: float,
    humidity: int,
    conditions: str,
    description: str,
    wind_speed: float,
    rain_1h: float = 0.0,
) -> str:
    """
    Persist one weather snapshot to ChromaDB.

    The 'document' string is what gets embedded (vectorised) — it's a
    natural-language description so semantic search works well.
    Structured numbers go into 'metadata' for fast filter queries.
    Returns the record ID.
    """
    ts = datetime.now(timezone.utc).isoformat()
    document = (
        f"{description}, temperature {temp:.1f} degrees celsius, "
        f"feels like {feels_like:.1f}, humidity {humidity} percent, "
        f"wind {wind_speed} meters per second"
        + (f", rain {rain_1h} mm in last hour" if rain_1h > 0 else "")
    )
    metadata = {
        "city": loc["city"],
        "region": loc.get("region", ""),
        "country": loc["country"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        "temp": temp,
        "feels_like": feels_like,
        "humidity": humidity,
        "conditions": conditions,        # e.g. "Rain", "Clear", "Clouds"
        "description": description,      # e.g. "light rain", "clear sky"
        "wind_speed": wind_speed,
        "rain_1h": rain_1h,
        "timestamp": ts,
        "date": ts[:10],
        "hour": int(ts[11:13]),
    }
    uid = f"{loc['city'].replace(' ', '_')}_{ts}"
    _col.add(documents=[document], metadatas=[metadata], ids=[uid])
    return uid


# ── MCP Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_current_weather() -> str:
    """
    Get current weather for your location (auto-detected via IP geolocation).
    Stores the result to ChromaDB history automatically.
    Requires OPENWEATHER_API_KEY env var.
    """
    if not OPENWEATHER_API_KEY:
        return (
            "OPENWEATHER_API_KEY is not set.\n"
            "Get a free key at https://openweathermap.org/api then run:\n"
            "  export OPENWEATHER_API_KEY=your_key_here"
        )

    loc = await _get_location()
    data = await _owm("weather", {"lat": loc["lat"], "lon": loc["lon"]})
    if not data:
        return "Failed to fetch weather from OpenWeatherMap. Check your API key."

    main   = data["main"]
    wind   = data["wind"]
    w      = data["weather"][0]
    rain   = data.get("rain", {}).get("1h", 0.0)
    clouds = data.get("clouds", {}).get("all", 0)
    vis    = data.get("visibility", None)

    # Store to ChromaDB — every call builds up history for analytics/predictions
    _store_record(
        loc,
        temp=main["temp"],
        feels_like=main["feels_like"],
        humidity=main["humidity"],
        conditions=w["main"],
        description=w["description"],
        wind_speed=wind["speed"],
        rain_1h=rain,
    )

    vis_str = f"{vis} m" if vis is not None else "N/A"
    return (
        f"Current Weather — {loc['city']}, {loc['region']}, {loc['country']}\n"
        f"{'━'*50}\n"
        f"Conditions    : {w['description'].title()}\n"
        f"Temperature   : {main['temp']}°C  (feels like {main['feels_like']}°C)\n"
        f"Humidity      : {main['humidity']}%\n"
        f"Wind          : {wind['speed']} m/s  dir {wind.get('deg', '?')}°\n"
        f"Pressure      : {main['pressure']} hPa\n"
        f"Cloud cover   : {clouds}%\n"
        f"Visibility    : {vis_str}\n"
        f"Rain (1h)     : {rain} mm\n"
        f"Coordinates   : {loc['lat']}, {loc['lon']}\n"
        f"{'─'*50}\n"
        f"[Saved to ChromaDB history — {_col.count()} records total]"
    )


@mcp.tool()
async def get_forecast(days: int = 5) -> str:
    """
    Get weather forecast for your location with rain probability.

    OWM free tier gives 3-hour slots for 5 days. We group them by day and
    show min/max temp, dominant conditions, and peak rain probability (pop).

    Args:
        days: How many days ahead (1–5, default 5)
    """
    if not OPENWEATHER_API_KEY:
        return "OPENWEATHER_API_KEY is not set."

    days = min(max(days, 1), 5)
    loc  = await _get_location()
    # cnt = slots to fetch; 8 slots/day × days
    data = await _owm("forecast", {"lat": loc["lat"], "lon": loc["lon"], "cnt": days * 8})
    if not data:
        return "Failed to fetch forecast."

    # Group 3-hour slots by calendar date
    daily: dict[str, list] = {}
    for slot in data["list"]:
        date = slot["dt_txt"][:10]
        daily.setdefault(date, []).append(slot)

    lines = [
        f"Forecast — {loc['city']}, {loc['country']}",
        "━" * 50,
    ]
    for date, slots in list(daily.items())[:days]:
        temps     = [s["main"]["temp"]         for s in slots]
        pops      = [s.get("pop", 0) * 100     for s in slots]   # pop = probability of precip (0–1)
        humidity  = mean([s["main"]["humidity"] for s in slots])
        conds     = [s["weather"][0]["description"] for s in slots]
        dominant  = max(set(conds), key=conds.count)
        rain_mm   = sum(s.get("rain", {}).get("3h", 0) for s in slots)

        lines += [
            f"\n{date}",
            f"  Temp        : {min(temps):.1f}°C → {max(temps):.1f}°C",
            f"  Conditions  : {dominant.title()}",
            f"  Rain chance : {max(pops):.0f}%",
            f"  Rain total  : {rain_mm:.1f} mm",
            f"  Humidity    : {humidity:.0f}%",
        ]

    return "\n".join(lines)


@mcp.tool()
async def get_weather_analytics() -> str:
    """
    Compute analytics over all weather history stored in ChromaDB.

    Stats include: temperature range/avg/trend, humidity, wind, rain frequency,
    top conditions, and a warming/cooling trend indicator.
    """
    result = _col.get(include=["metadatas"])
    metas  = result["metadatas"]

    if not metas:
        return (
            "No history yet. Call get_current_weather a few times to build up data,\n"
            "then come back for analytics."
        )

    # Extract numeric series
    temps      = [m["temp"]       for m in metas]
    humidities = [m["humidity"]   for m in metas]
    winds      = [m["wind_speed"] for m in metas]
    rains      = [m["rain_1h"]    for m in metas]
    conds      = [m["conditions"] for m in metas]
    dates      = sorted(set(m["date"] for m in metas))

    # Condition frequency table
    freq: dict[str, int] = {}
    for c in conds:
        freq[c] = freq.get(c, 0) + 1
    top3 = sorted(freq.items(), key=lambda x: -x[1])[:3]

    # Temperature trend — split history in half, compare means
    mid        = len(temps) // 2
    trend_str  = "→ stable"
    if mid >= 1:
        diff = mean(temps[mid:]) - mean(temps[:mid])
        if   diff >  1.5: trend_str = f"↑ warming  (+{diff:.1f}°C across history)"
        elif diff < -1.5: trend_str = f"↓ cooling  ({diff:.1f}°C across history)"

    temp_sd    = stdev(temps) if len(temps) > 1 else 0
    wind_sd    = stdev(winds) if len(winds) > 1 else 0
    rain_count = sum(1 for r in rains if r > 0)
    rain_pct   = rain_count / len(rains) * 100

    return (
        f"Weather Analytics\n"
        f"{'━'*50}\n"
        f"Records       : {len(metas)}  ({dates[0]} → {dates[-1]})\n"
        f"\nTemperature\n"
        f"  Min / Max   : {min(temps):.1f}°C / {max(temps):.1f}°C\n"
        f"  Average     : {mean(temps):.1f}°C  (σ={temp_sd:.1f}°C  "
        f"{'stable' if temp_sd < 3 else 'variable'})\n"
        f"  Trend       : {trend_str}\n"
        f"\nHumidity\n"
        f"  Avg / Max   : {mean(humidities):.0f}% / {max(humidities)}%\n"
        f"\nWind\n"
        f"  Avg / Max   : {mean(winds):.1f} m/s / {max(winds):.1f} m/s  (σ={wind_sd:.1f})\n"
        f"\nRain\n"
        f"  Rainy checks: {rain_count}/{len(metas)}  ({rain_pct:.0f}%)\n"
        f"  Total logged: {sum(rains):.1f} mm\n"
        f"\nTop Conditions\n"
        + "\n".join(f"  {c:20s}: {n}x  ({n/len(metas)*100:.0f}%)" for c, n in top3)
    )


@mcp.tool()
async def search_weather_history(query: str, top_k: int = 5) -> str:
    """
    Semantic vector search over stored weather history using ChromaDB.

    The query is embedded with the same model used at insert time, and the
    nearest neighbours (by cosine distance) are returned.  Great for questions
    like 'rainy days', 'hot and humid afternoons', 'stormy weather'.

    Args:
        query: Natural language description of weather to search for
        top_k: Number of results to return (default 5)
    """
    total = _col.count()
    if total == 0:
        return "No history yet. Run get_current_weather to start collecting data."

    top_k   = min(top_k, total)
    results = _col.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    dists  = results["distances"][0]

    lines = [f"Search: \"{query}\"  ({top_k} closest matches from {total} records)", "━"*50]
    for doc, meta, dist in zip(docs, metas, dists):
        # ChromaDB returns L2 or cosine distance; convert to a rough % similarity
        similarity = max(0.0, (1.0 - dist) * 100)
        lines += [
            f"\n{meta['date']}  {meta['hour']:02d}:00 — {meta['city']}, {meta['country']}",
            f"  {meta['description'].title()}  |  {meta['temp']:.1f}°C  |  "
            f"humidity {meta['humidity']}%  |  wind {meta['wind_speed']} m/s",
            f"  Similarity: {similarity:.0f}%",
        ]
    return "\n".join(lines)


@mcp.tool()
async def get_weather_prediction() -> str:
    """
    Predict upcoming weather based on patterns in ChromaDB history.

    Method:
      - Temperature: ordinary least-squares linear trend extrapolated one step
      - Rain: frequency in the most recent 10 records
      - Conditions: mode of recent conditions
      - Humidity: average of last 5 records

    More history = better predictions.  Combine with get_forecast for
    a fuller picture.
    """
    result = _col.get(include=["metadatas"])
    metas  = result["metadatas"]

    if len(metas) < 3:
        return (
            f"Need at least 3 records for prediction (have {len(metas)}).\n"
            "Run get_current_weather a few times first."
        )

    # Sort chronologically
    ordered = sorted(metas, key=lambda m: m["timestamp"])

    temps  = [m["temp"]    for m in ordered]
    n      = len(temps)

    # ── Linear regression slope on temperature ───────────────────────────────
    # slope = Σ (i - ī)(t - t̄) / Σ (i - ī)²
    x_mean = (n - 1) / 2.0
    y_mean = mean(temps)
    denom  = sum((i - x_mean) ** 2 for i in range(n))
    slope  = (
        sum((i - x_mean) * (temps[i] - y_mean) for i in range(n)) / denom
        if denom > 0 else 0.0
    )
    predicted_next = temps[-1] + slope

    # ── Rain probability from recent 10 records ──────────────────────────────
    recent       = ordered[-min(10, n):]
    rain_hits    = sum(1 for m in recent if m["rain_1h"] > 0)
    rain_pct     = rain_hits / len(recent) * 100

    # ── Likely condition ─────────────────────────────────────────────────────
    recent_conds = [m["conditions"] for m in recent]
    likely_cond  = max(set(recent_conds), key=recent_conds.count)

    # ── Humidity (last 5) ────────────────────────────────────────────────────
    avg_humidity = mean(m["humidity"] for m in ordered[-5:])

    direction = "warming" if slope > 0.2 else "cooling" if slope < -0.2 else "stable"
    rain_icon = "⛈  Yes" if rain_pct > 40 else "🌦  Possible" if rain_pct > 20 else "☀  Unlikely"

    return (
        f"Weather Prediction  (from {n} historical records)\n"
        f"{'━'*50}\n"
        f"Temperature\n"
        f"  Last recorded   : {temps[-1]:.1f}°C\n"
        f"  Predicted next  : {predicted_next:.1f}°C\n"
        f"  Trend           : {direction}  (slope {slope:+.3f}°C / interval)\n"
        f"\nRain\n"
        f"  Recent frequency: {rain_hits}/{len(recent)} checks  ({rain_pct:.0f}%)\n"
        f"  Rain likely?    : {rain_icon}\n"
        f"\nConditions        : {likely_cond}\n"
        f"Avg Humidity      : {avg_humidity:.0f}%\n"
        f"\nTip: call get_forecast for a data-backed 5-day outlook too."
    )
