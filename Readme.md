# Spotify Song Recommendation Project

## Project Description

This project builds a playlist-based song recommendation system using the Spotify Million Playlist Dataset (MPD). The pipeline extracts playlist-track interactions from raw JSON, prepares model-ready data, and supports item-based nearest-neighbor recommendation workflows. User inputs a playlist, and the system will recommend songs that go with the existing songs (seed tracks) in the playlist.


## Dataset Description

The dataset is stored as MPD JSON slices under [dataset/data](dataset/data), with each slice containing approximately 1000 playlists, for example [dataset/data/mpd.slice.0-999.json](dataset/data/mpd.slice.0-999.json). The project primarily uses playlist and track metadata fields such as `pid`, `track_uri`, `track_name`, `artist_name`, `album_name`, and `album_uri`.

```json
{
	"playlists": [
		{
			"pid": 0,
			"num_followers": 1,
			"tracks": [
				{
					"track_uri": "spotify:track:...",
					"track_name": "...",
					"artist_name": "...",
					"album_name": "...",
					"album_uri": "spotify:album:..."
				}
			]
		}
	]
}
```

## Building the ETL pipeline

The initial ETL approach loaded JSON slices into memory and wrote flattened CSV output, which was simple but slow at scale. The current ETL in [src/track_etl.py](src/track_etl.py) uses generators for streaming extraction, applies lightweight track normalization, and writes Parquet outputs for better I/O performance. The pipeline was then parallelized in [src/train.py](src/train.py) using `ProcessPoolExecutor` on MacBook M3 cores, reducing end-to-end ETL time from about 277 seconds to about 50 seconds while processing roughly 6 million tracks across 1 million playlists.

Additionally, using Apache Spark for the ETL step (instead of pandas/ProcessPoolExecutor) saved about 5 more seconds on the same workload


## Data Cleaning / Preparation

Cleaning includes safe JSON parsing, field selection, default handling for missing metadata, URI normalization by stripping Spotify prefixes, and Parquet output writing with automatic output-directory creation. The extracted dataset currently contains `playlist_id`, `track_uri`, `artist_name`, `track_name`, `album_name`, and `album_uri`.


## Model Comparison

All models use 64 latent factors where applicable. Inference is measured as a 5-track seed query returning top-10 recommendations. ALS and BPR use the `implicit` library (Python 3.11) and run on the full dataset.

| Model | Dataset | Training Time | Inference Time | Notes |
|---|---|---|---|---|
| KNN (brute-force cosine) | 1M playlists × 2.26M tracks | 0.15 s | ~20 s | No real training; cost deferred entirely to query time |
| ALS (implicit feedback) | 1M playlists × 2.26M tracks | 792 s | 4.60 s | `implicit` ALS, 3 iterations, 64 factors |
| BPR (pairwise SGD) | 1M playlists × 2.26M tracks | 466 s | 21.3 s | `implicit` BPR, 5 epochs, 64 factors, final AUC 94.1% |
| Track2Vec (SVD proxy) | 1M playlists × 2.26M tracks | 139 s | 2.63 s | TruncatedSVD, 64 components — approximates skip-gram SGNS |

## General Code Running Guidelines

Install dependencies (Python 3.14):

```bash
pip install -r requirements.txt
```

ALS and BPR require the `implicit` library, which needs Python 3.11 (no pre-built wheels for 3.14). Install it once:

```bash
py -3.11 -m pip install implicit
```

Run ETL:

```bash
python -m src.track_etl
```

Run benchmark:

```bash
python -m src.utils.benchmark
```

Run parallel ETL orchestration:

```bash
python -m src.train
```

Run model training and benchmarks:

```bash
python -m src.train_knn
py -3.11 -m src.train_als
py -3.11 -m src.train_bpr
python -m src.train_track2vec
```

Run inference:

```bash
python -m src.inference
```

Run tests:

```bash
python -m pytest -q
```
