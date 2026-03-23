# Weather MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives Claude live weather data, forecasts, and an analytics layer backed by a persistent vector database.

Built with:
- **OpenWeatherMap API** — live weather + 5-day forecasts
- **ipinfo.io** — zero-config location detection from your IP
- **ChromaDB** — persistent vector store for history, semantic search, and trend analysis
- **FastMCP** — minimal boilerplate MCP server framework
- **XGBoost** — trained ML model for weather prediction (optional, see below)

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
| `get_weather_prediction` | XGBoost ML prediction (auto-fallback to linear regression if model not trained yet) |

---

## Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) — fast Python package manager
- [Claude Code CLI](https://claude.ai/claude-code)
- A free [OpenWeatherMap API key](https://openweathermap.org/api)

For ML training (optional):
- `pandas`, `matplotlib`, `scikit-learn`, `requests` — only needed to run the training notebook

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

### Prediction method — XGBoost (trained) or linear fallback

When `models/weather_xgb_temp.json` and `models/weather_xgb_rain.json` exist the server
uses two XGBoost models trained on Open-Meteo historical data:

| Model | Type | Target |
|---|---|---|
| `weather_xgb_temp.json` | `XGBRegressor` | Next observation temperature (°C) |
| `weather_xgb_rain.json` | `XGBClassifier` | Rain probability (0–1) |

**Feature vector (10 features, order fixed):**

| # | Feature | Source at inference |
|---|---|---|
| 1 | `hour` | Current UTC hour |
| 2 | `month` | Current month |
| 3 | `day_of_year` | Current day of year |
| 4 | `temp_lag1` | Most recent ChromaDB record |
| 5 | `temp_lag2` | Second most recent |
| 6 | `temp_lag3` | Third most recent |
| 7 | `humidity` | Most recent record |
| 8 | `wind_speed` | Most recent record |
| 9 | `pressure` | Most recent record |
| 10 | `rain_lag1` | Most recent record |

When models are absent the tool falls back to:
- **Temperature**: OLS linear regression extrapolated one step
- **Rain**: fraction of last 10 records with `rain_1h > 0`

---

## Project structure

```
weather/
├── weather.py                   # MCP server — all tools and ChromaDB logic
├── main.py                      # Entrypoint — runs mcp.run(transport="stdio")
├── train_weather_model.ipynb    # Notebook: fetch data, train, save XGBoost models
├── models/
│   ├── weather_xgb_temp.json    # XGBRegressor  (generated by notebook)
│   ├── weather_xgb_rain.json    # XGBClassifier (generated by notebook)
│   └── feature_names.json       # Feature order reference
├── pyproject.toml               # Dependencies and project metadata
├── uv.lock                      # Locked dependency versions
├── .python-version              # Python version pin (3.14)
└── README.md
```

ChromaDB data is stored outside the repo at `~/.weather_mcp/db`.
Model `.json` files are git-ignored (regenerate via the notebook).

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

## Training the XGBoost models

```bash
# Install notebook dependencies (once)
uv add pandas matplotlib scikit-learn requests

# Launch Jupyter and open the notebook
uv run jupyter notebook train_weather_model.ipynb
```

Run all cells top-to-bottom (~1–2 minutes):
1. Downloads 2 years × 5 cities of hourly Open-Meteo data (free, no API key)
2. Engineers lag and time features
3. Trains `XGBRegressor` (temperature) and `XGBClassifier` (rain)
4. Saves models to `models/`

After training, restart the MCP server and `get_weather_prediction` will automatically
switch from the linear fallback to the XGBoost model.

**Expected accuracy on held-out test set:**
- Temperature MAE ≈ 0.8–1.2 °C
- Rain AUC-ROC ≈ 0.85–0.92

---

## Limitations

- Location is IP-based — may not be accurate if you're on a VPN or corporate network
- OpenWeatherMap free tier has a 60 calls/minute limit
- Forecast rain probability (`pop`) is per 3-hour slot, aggregated to daily max
- XGBoost models trained on 5 global cities — accuracy improves if retrained on local data
- Only works with metric units (°C, m/s)
