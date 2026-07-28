# Collaborative Playlist Recommendation System

Processes the Spotify Million Playlist Dataset (MPD) through an Apache Spark ETL pipeline, then serves analytics and recommendation data from a single FastAPI app.

---

## Project Structure

```
backend/          Python package — ETL pipeline + FastAPI
  etl/            Spark + pandas ETL logic
  api/            FastAPI application
    templates/    Jinja2 templates (server-rendered dashboard)
    static/       CSS served by the FastAPI app
  utils/          Shared config, logging, Spark session, schema
  tests/          pytest test suite
dataset/
  mpd.example.json  Small bundled sample slice (tracked in git, used by default)
  data/             Full MPD JSON slices (not tracked in git; download separately)
output/           ETL outputs (not tracked in git)
hadoop/           Windows winutils for local Spark (not tracked in git)
```

---

## Dataset

The [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-millionsong-dataset-challenge) is distributed as ~1000 JSON slices, each containing 1000 playlists:

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

By default, `backend/config.yaml` points `dataset_path` at the small bundled `dataset/mpd.example.json` sample (3 playlists) so the pipeline and API work out of the box without downloading anything. Once you've downloaded the full dataset into `dataset/data/`, update `dataset_path` in `backend/config.yaml` to `"dataset/data"`.

---

## ETL Pipeline

Run once to populate `output/` before starting the API.

```bash
python -m backend.run_etl
```

The pipeline reads the configured JSON slice(s) with Spark, flattens nested playlists/tracks, strips Spotify URI prefixes, and writes:

| File | Description |
|------|-------------|
| `output/tracks.parquet` | Flattened, cleaned track records |
| `output/id_mappings.parquet` | Stable `track_uri → track_id` (contiguous ints for models) |
| `output/stats/track_counts.parquet` | Track appearance frequency across playlists |
| `output/stats/artist_counts.parquet` | Artist appearance frequency |
| `output/stats/playlist_sizes.parquet` | Per-playlist track counts |
| `output/stats.json` | Same summary stats shown by the dashboard, as static JSON |
| `output/interaction_matrix.npz` | Sparse playlist–track COO matrix for collaborative filtering |

ETL history: started as in-memory CSV, moved to streaming Parquet generators, parallelized with `ProcessPoolExecutor` (~277s → ~50s on M3), then migrated to Apache Spark.

---

## Running the App

```bash
uvicorn backend.api.main:app --reload
```

Open `http://localhost:8000` for the dashboard (stat cards, top-tracks list, top-artists list), rendered server-side with Jinja2 from the ETL output. If the ETL hasn't been run yet, the page shows a notice instead of erroring.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Server-rendered dashboard |
| `GET /api/stats?top_n=10` | Summary counts + top tracks and artists |
| `GET /api/top-tracks?limit=20` | Ranked track list |
| `GET /api/top-artists?limit=20` | Ranked artist list |
| `GET /api/tracks?page=0&size=50` | Paginated full track table |

Interactive API docs: `http://localhost:8000/docs`

---

## Setup

**API only** (no ETL, no PySpark) — dashboard + JSON endpoints:

```bash
pip install -r backend/api/requirements.txt
python -m backend.run_etl   # uses dataset/mpd.example.json by default
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
