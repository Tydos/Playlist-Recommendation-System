# Spotify Song Recommendation Project

## Project Description

This project processes the Spotify Million Playlist Dataset (MPD) and converts raw JSON playlist data into a clean, structured format suitable for downstream recommendation models and analytics.


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

The initial ETL approach loaded JSON slices into memory and wrote flattened CSV output, which was simple but slow at scale. A better version used generators for streaming extraction, applies lightweight track normalization, and writes Parquet outputs for better I/O performance. The pipeline was then parallelized using `ProcessPoolExecutor` on MacBook M3 cores, reducing end-to-end ETL time from about 277 seconds to about 50 seconds while processing roughly 6 million tracks across 1 million playlists. The current pipeline usses Apache Spark.


## Data Cleaning / Preparation

Cleaning includes safe JSON parsing, field selection, default handling for missing metadata, URI normalization by stripping Spotify prefixes, and Parquet output writing with automatic output-directory creation. The extracted dataset currently contains `playlist_id`, `track_uri`, `artist_name`, `track_name`, `album_name`, and `album_uri`.


## General Code Running Guidelines

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ETL:

```bash
python -m src.track_etl
```

Run tests:

```bash
python -m pytest -q
```
