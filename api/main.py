from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.popularity import PopularityRecommender
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger("api")

_data: dict[str, pd.DataFrame] = {}
_API_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_API_DIR / "templates"))

def _output_dir() -> Path:
    config = load_config()
    return Path(config.get("output_path", "output"))


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}. Run the ETL first.")
    return pd.read_parquet(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    out = _output_dir()
    _data["tracks"] = _load_parquet(out / "tracks.parquet")
    _data["track_counts"] = _load_parquet(out / "stats" / "track_counts.parquet")
    _data["artist_counts"] = _load_parquet(out / "stats" / "artist_counts.parquet")
    _data["playlist_sizes"] = _load_parquet(out / "stats" / "playlist_sizes.parquet")
    logger.info(f"Loaded output data from {out}")
    yield


app = FastAPI(title="Playlist Recommendation System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_API_DIR / "static")), name="static")


def _build_stats(top_n: int = 10) -> dict[str, Any]:
    track_counts = _data["track_counts"]
    artist_counts = _data["artist_counts"]
    playlist_sizes = _data["playlist_sizes"]

    top_tracks = (
        track_counts.head(top_n)[["track_name", "artist_name", "count"]]
        .to_dict(orient="records")
    )
    top_artists = (
        artist_counts.head(top_n)[["artist_name", "count"]]
        .to_dict(orient="records")
    )

    return {
        "total_tracks": len(track_counts),
        "total_artists": len(artist_counts),
        "total_playlists": len(playlist_sizes),
        "avg_playlist_size": round(float(playlist_sizes["track_count"].mean()), 1),
        "top_tracks": top_tracks,
        "top_artists": top_artists,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    stats = _build_stats(top_n=10)
    seed_tracks = _data["track_counts"].head(3)["track_uri"].tolist()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"loaded": True, "stats": stats, "seed_tracks": seed_tracks},
    )


@app.get("/api/stats")
def get_stats(top_n: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    return _build_stats(top_n=top_n)


@app.get("/api/top-tracks")
def get_top_tracks(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    tc = _data["track_counts"].head(limit).reset_index(drop=True)
    tc["rank"] = tc.index + 1
    return tc[["rank", "track_name", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/top-artists")
def get_top_artists(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    ac = _data["artist_counts"].head(limit).reset_index(drop=True)
    ac["rank"] = ac.index + 1
    return ac[["rank", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/tracks")
def get_tracks(
    page: int = Query(0, ge=0),
    size: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    tracks = _data["tracks"]
    cols = ["playlist_id", "track_uri", "track_name", "artist_name", "album_name"]
    subset = tracks[cols].iloc[page * size : (page + 1) * size]
    return {
        "total": len(tracks),
        "page": page,
        "size": size,
        "data": subset.to_dict(orient="records"),
    }

@app.get("/api/tracks/search")
def search_tracks(
    q: str = Query(..., min_length=1, description="Song name substring to search for"),
    limit: int = Query(8, ge=1, le=50),
) -> list[dict]:
    tc = _data["track_counts"]
    matches = tc[tc["track_name"].str.contains(q, case=False, na=False, regex=False)]
    return matches.head(limit)[["track_uri", "track_name", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/recommend")
def recommend(
    seed_tracks: list[str] = Query(..., description="List of track URIs/Spotify URI"),
    limit: int = Query(20, ge=1, le=200),
) -> list[dict]:
    return PopularityRecommender(_data["track_counts"]).recommend(seed_tracks, limit=limit)