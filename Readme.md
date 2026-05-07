# Collaborative Playlist Recommendation System

Processes the Spotify Million Playlist Dataset (MPD) through an Apache Spark ETL pipeline, then serves analytics and recommendation data via a FastAPI backend and a Vite React dashboard.

---

## Project Structure

```
backend/          Python package — ETL pipeline + FastAPI
  etl/            Spark + pandas ETL logic
  api/            FastAPI application
  utils/          Shared config, logging, Spark session, schema
  tests/          pytest test suite
frontend/         Vite + React dashboard (Recharts)
dataset/data/     MPD JSON slices (not tracked in git)
output/           ETL outputs (not tracked in git)
hadoop/           Windows winutils for local Spark (not tracked in git)
```

---

## Dataset

The [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-millionsong-dataset-challenge) is stored as ~1000 JSON slices under `dataset/data/`, each containing 1000 playlists.

```json
{
  "playlists": [{
    "pid": 0,
    "tracks": [{
      "track_uri": "spotify:track:...",
      "track_name": "...",
      "artist_name": "...",
      "album_name": "...",
      "album_uri": "spotify:album:...",
      "artist_uri": "spotify:artist:..."
    }]
  }]
}
```

---

## ETL Pipeline

Run once to populate `output/` before starting the API.

```bash
python -m backend.run_etl
```

The pipeline reads all JSON slices with Spark, flattens nested playlists/tracks, strips Spotify URI prefixes, and writes four outputs:

| File | Description |
|------|-------------|
| `output/tracks.parquet` | Flattened, cleaned track records |
| `output/id_mappings.parquet` | Stable `track_uri → track_id` (contiguous ints for models) |
| `output/stats/track_counts.parquet` | Track appearance frequency across playlists |
| `output/stats/artist_counts.parquet` | Artist appearance frequency |
| `output/stats/playlist_sizes.parquet` | Per-playlist track counts |
| `output/interaction_matrix.npz` | Sparse playlist–track COO matrix for collaborative filtering |

ETL history: started as in-memory CSV, moved to streaming Parquet generators, parallelized with `ProcessPoolExecutor` (~277s → ~50s on M3), then migrated to Apache Spark.

---

## Backend API

```bash
uvicorn backend.api.main:app --reload
```

Runs at `http://localhost:8000`. Loads output parquets at startup.

| Endpoint | Description |
|----------|-------------|
| `GET /api/stats?top_n=10` | Summary counts + top tracks and artists |
| `GET /api/top-tracks?limit=20` | Ranked track list |
| `GET /api/top-artists?limit=20` | Ranked artist list |
| `GET /api/tracks?page=0&size=50` | Paginated full track table |

Interactive docs: `http://localhost:8000/docs`

---

## Frontend Dashboard

```bash
cd frontend
npm install      # first time only
npm run dev      # → http://localhost:5173
```

Shows four stat cards (unique tracks, artists, playlists, avg playlist size), a top-tracks table, and a top-artists bar chart. Reads from the API at `http://localhost:8000`.

---

## Setup

**API only** (no ETL, no PySpark):

```bash
pip install -r backend/api/requirements.txt
uvicorn backend.api.main:app --reload
```

**Full local dev** (ETL + API):

```bash
pip install -r requirements.txt
```

> Requires Java 8+ and Hadoop `winutils` on Windows for PySpark. See `hadoop/` directory.

---

## Tests

```bash
pytest -q
```
