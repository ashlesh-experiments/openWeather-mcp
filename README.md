# Weather MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives Claude live weather data, forecasts, and an analytics layer backed by a persistent vector database.

Built with:
- **OpenWeatherMap API** — live weather + 5-day forecasts
- **ipinfo.io** — zero-config location detection from your IP
- **ChromaDB** — persistent vector store for history, semantic search, and trend analysis
- **FastMCP** — minimal boilerplate MCP server framework

---

## What it does

Every time you ask Claude about the weather, the server:
1. Detects your location automatically (no input needed)
2. Fetches live data from OpenWeatherMap
3. Stores the result in ChromaDB on disk (`~/.weather_mcp/db`)
4. Over time, builds up a history you can query for analytics and predictions

---

## Tools

| Tool | What Claude uses it for |
|---|---|
| `get_current_weather` | Current conditions — temperature, humidity, wind, rain, pressure |
| `get_forecast` | 5-day / 3-hour forecast with rain probability (`pop`) per day |
| `get_weather_analytics` | Stats from stored history: temp range/trend, rain frequency, top conditions |
| `search_weather_history` | Semantic vector search — "find past stormy days", "hot and humid" |
| `get_weather_prediction` | Linear trend extrapolation + rain probability from recent history |

---

## Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) — fast Python package manager
- [Claude Code CLI](https://claude.ai/claude-code)
- A free [OpenWeatherMap API key](https://openweathermap.org/api)

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone <repository-url>
cd weather
uv sync
```

`uv sync` creates a virtual environment and installs everything from `uv.lock`.
On first run, ChromaDB will also download a ~30 MB ONNX embedding model for semantic search.

**2. Set your OpenWeatherMap API key**
```bash
cp .env.example .env

visit : [https://home.openweathermap.org/] and create api key and paste it in .env

update .env

```
**3. Register with Claude Code**
```bash
claude mcp add weather -- uv --directory /path/to/weather run python main.py
```

Verify it's connected:
```bash
claude mcp list
# weather: uv --directory ... run python main.py - ✓ Connected
```

---

## Usage

Just talk to Claude naturally — no special commands needed.

**Current weather**
> "What's the weather like right now?"
> "How's the weather today?"

**Forecast**
> "Give me a 5-day forecast"
> "Will it rain this week?"
> "What's the weather looking like for the next 3 days?"

**Analytics (requires history)**
> "Show me my weather history analytics"
> "What's the temperature trend been lately?"
> "How often has it rained recently?"

**Semantic search over history**
> "Find past days when it was stormy"
> "When was the last time it was hot and humid?"
> "Search my history for rainy afternoons"

**Prediction**
> "Predict tomorrow's weather based on past data"
> "What does the temperature trend suggest?"

> **Note:** Analytics, search, and predictions get better the more you use it.
> Call `get_current_weather` regularly to build up history.

---

## How it works internally

### Location detection
`ipinfo.io/json` returns `{"loc": "lat,lon", "city": ..., "country": ...}` from your outbound IP. No API key required. Called automatically on every request.

### OpenWeatherMap endpoints
| Endpoint | Data |
|---|---|
| `GET /data/2.5/weather` | Current conditions, `rain.1h` field |
| `GET /data/2.5/forecast` | 3-hour slots × 5 days, `pop` = probability of precipitation (0–1) |

All requests use `units=metric`.

### ChromaDB storage schema
Each weather check is stored as one document:

```
document (embedded):  "light rain, temperature 18.2 degrees celsius, feels like 16.5, humidity 82 percent, wind 3.1 meters per second"

metadata: {
  city, region, country, lat, lon,
  temp, feels_like, humidity,
  conditions,   # "Rain" / "Clear" / "Clouds" etc.
  description,  # "light rain" / "clear sky" etc.
  wind_speed, rain_1h,
  timestamp, date, hour
}
```

The document text is vectorised by a small ONNX sentence-transformer model. This enables semantic similarity search — ChromaDB finds records whose embedded vector is closest to your query's vector, so "stormy weather" matches "thunderstorm" and "heavy rain" without exact keywords.

### Prediction method
- **Temperature trend**: ordinary least-squares linear regression over all stored temps, extrapolated one step forward
- **Rain probability**: fraction of the last 10 records that had `rain_1h > 0`
- **Likely conditions**: mode of conditions in the last 10 records

---

## Project structure

```
weather/
├── weather.py        # MCP server — all tools and ChromaDB logic
├── main.py           # Entrypoint — runs mcp.run(transport="stdio")
├── pyproject.toml    # Dependencies and project metadata
├── uv.lock           # Locked dependency versions
├── .python-version   # Python version pin (3.14)
└── README.md
```

ChromaDB data is stored outside the repo at `~/.weather_mcp/db`.

---

## Dependency management

```bash
# Add a package
uv add <package>

# Remove a package
uv remove <package>

# Upgrade all dependencies
uv lock --upgrade && uv sync

# Export requirements.txt (for legacy tooling)
uv export --format requirements-txt --output-file requirements.txt
```

---

## Limitations

- Location is IP-based — may not be accurate if you're on a VPN or corporate network
- OpenWeatherMap free tier has a 60 calls/minute limit
- Forecast rain probability (`pop`) is per 3-hour slot, aggregated to daily max
- Predictions use simple linear regression — not a trained ML model
- Only works with metric units (°C, m/s)
