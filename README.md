# Collaborative Playlist Recommendation System

This project studies real Spotify playlists to understand what music people listen to together. It lays the groundwork for recommending songs based on those patterns.

It works in two stages:

1. **ETL (Extract, Transform, Load).** Reads raw playlist data and organizes it into clean, analyzable files
2. **Dashboard & API.** Displays insights like most popular songs and artists

---

## Project Structure

```
etl/            Spark ETL: extract, transform, load, build stats + interaction matrix
  run_etl.py    ETL entry point
api/
  main.py       FastAPI app and routes
  templates/    Jinja2 templates (server-rendered dashboard)
  static/       CSS served by the FastAPI app
utils/          Shared config, logging, Spark session, schema, S3 upload
tests/          pytest test suite
config.yaml     Dataset paths, output paths, Spark tuning
dataset/
  mpd.example.json  Small bundled sample slice (tracked in git, used by default)
  data/             Full MPD JSON slices (not tracked in git; download separately)
output/           ETL outputs (not tracked in git)
hadoop/           Windows winutils for local Spark (not tracked in git)
pyproject.toml    Project metadata, dependencies, ruff config
README.md
```

The project is a single Python service. Everything, ETL and the dashboard, runs from the project root, with no separate frontend build or Node toolchain required.

---

## How the ETL Process Works

### Step-by-step

1. **Extract.** Read JSON playlist files from the dataset folder
2. **Transform.** Flatten nested playlists into rows, clean up song IDs, assign unique playlist numbers
3. **Build lookups.** Create ID tables for songs and playlists
4. **Compute stats.** Count how often songs and artists appear
5. **Build interaction matrix.** Map which songs appear in which playlists
6. **Load.** Save everything to the `output/` folder

### Run the ETL

```bash
python -m etl.run_etl
```

Run this once before starting the dashboard. It populates the `output/` folder.

### ETL evolution (naive vs optimal)

The pipeline went through three stages as the dataset grew:

| Approach | How it works | Trade-off |
|----------|--------------|-----------|
| **Naive** | Load each JSON slice fully into memory, flatten to rows, write CSV | Simple to build, but slow and memory-heavy at scale |
| **Optimized (pandas)** | Stream rows with generators, normalize URIs, write Parquet per slice, parallelize across cores with `ProcessPoolExecutor` | ~277s → ~50s on M3 for ~6M track rows across 1M playlists |
| **Current (Spark)** | Distributed JSON read, flatten playlists/tracks in Spark, write Parquet, then pandas/scipy for stats and the interaction matrix | Best fit for the full MPD; saves a further ~5s over the parallel pandas path on the same workload |

**Naive.** Read entire JSON files into memory, process sequentially, output CSV. Works for small samples but does not scale.

**Optimized.** Switch to generator-based extraction (one row at a time), Parquet output (columnar, compressed), and `ProcessPoolExecutor` to process multiple slices in parallel. This cut end-to-end ETL from ~277 seconds to ~50 seconds.

**Current.** Apache Spark handles the heavy JSON parsing and flattening across partitions. Post-processing (ID mappings, stats, interaction matrix) still runs in pandas/scipy on the written Parquet files. See `etl/track_etl.py`.

---

## Input: What the Raw Data Looks Like

The project uses the [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-millionsong-dataset-challenge). About **1 million real playlists** that Spotify users created.

### How it's organized

The dataset comes as **~1,000 JSON files** (called "slices"). Each file contains **1,000 playlists**, and each playlist contains a list of songs.

```
dataset/
  mpd.example.json          ← small sample (3 playlists) for testing
  data/
    mpd.slice.0-999.json    ← 1,000 playlists
    mpd.slice.1000-1999.json ← another 1,000 playlists
    ... (about 1,000 files total)
```

### What one playlist looks like inside a JSON file

Each file is structured like a folder of playlists, where each playlist has a list of songs:

```json
{
  "playlists": [
    {
      "name": "Throwback Party",
      "pid": 0,
      "tracks": [
        {
          "track_uri": "spotify:track:1301WleyT98MSxVHPZCA6M",
          "track_name": "HUMBLE.",
          "artist_name": "Kendrick Lamar",
          "album_name": "DAMN."
        },
        {
          "track_uri": "spotify:track:7BKLCZ1jbUBVqRi2FVlTVw",
          "track_name": "Closer",
          "artist_name": "The Chainsmokers",
          "album_name": "Collage EP"
        }
      ]
    }
  ]
}
```

**Important detail:** `pid` is only unique *within one file*. Playlist #0 in file A and playlist #0 in file B are different playlists. The ETL handles this by assigning each playlist a globally unique `playlist_id`.

### Configuration

In `config.yaml`:

```yaml
dataset_path: "dataset/mpd.example.json"   # small sample (default)
# dataset_path: "dataset/data"            # full dataset (after download)
```

---

## Output: What Gets Produced

After ETL runs, the `output/` folder contains organized data files:

| File | Example | Used for |
|------|---------|----------|
| `tracks.parquet` | `slice_id=mpd.slice.0-999.json`, `pid=0`, `playlist_id=0`, `track_name="HUMBLE."`, `artist_name="Kendrick Lamar"` | Every song-in-playlist occurrence; feeds stats and the interaction matrix |
| `id_mappings.parquet` | `track_uri=1301WleyT98MSxVHPZCA6M`, `track_name="HUMBLE."`, `track_id=0` | Unique songs with simple numeric IDs for the recommendation model |
| `playlist_mappings.parquet` | `slice_id=mpd.slice.0-999.json`, `pid=0`, `playlist_id=0` | Unique playlists with global IDs; prevents cross-slice collisions |
| `interaction_matrix.npz` | Playlist 0 → HUMBLE.=1, Closer=1; Playlist 1 → HUMBLE.=1, Starboy=1 | Sparse playlist × song grid for collaborative filtering |
| `stats/track_counts.parquet` | `track_name="One Dance"`, `artist_name="Drake"`, `count=98` | Song popularity rankings for the dashboard |
| `stats/artist_counts.parquet` | `artist_name="Drake"`, `count=1929` | Artist popularity rankings for the dashboard |
| `stats/playlist_sizes.parquet` | `playlist_id=0`, `track_count=52` | Playlist length stats (e.g. average playlist size) |
| `stats.json` | `total_playlists=2000`, `avg_playlist_size=67.1`, `top_tracks=[...]` | Dashboard-ready summary snapshot |

---

## Running the App

```bash
uvicorn api.main:app --reload
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

Dependencies are declared in `pyproject.toml`. The base install covers the API and dashboard; ETL (PySpark) and S3 backup are optional extras.

**API only** (no ETL, no PySpark). Serve an already-populated `output/` folder:

```bash
pip install -e .
uvicorn api.main:app --reload
```

**Full local dev** (ETL + API + S3 backup + tests/lint):

```bash
pip install -e ".[all]"
python -m etl.run_etl   # uses dataset/mpd.example.json by default
uvicorn api.main:app --reload
```

> Requires Java 8+ and Hadoop `winutils` on Windows for PySpark. See `hadoop/` directory.

---

## Tests

```bash
pytest -q
```
