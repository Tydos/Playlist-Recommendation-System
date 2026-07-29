from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.cooccurence import ItemItemRecommender
from models.popularity import PopularityRecommender
from utils.data_loader import app_data, load_etl_output
from utils.pydantic_schemas import RecommendationScore
from utils.logging import get_logger
from utils.stats import build_stats

logger = get_logger("api")

_API_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_API_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data.update(load_etl_output())
    logger.info("Loaded output data")
    yield

app = FastAPI(title="Playlist Recommendation System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_API_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    stats = build_stats(
        app_data["track_counts"],
        app_data["artist_counts"],
        app_data["playlist_sizes"],
        top_n=10,
    )
    seed_tracks = app_data["track_counts"].head(3)["track_uri"].tolist()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"loaded": True, "stats": stats, "seed_tracks": seed_tracks},
    )


@app.get("/api/stats")
def get_stats(top_n: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    return build_stats(
        app_data["track_counts"],
        app_data["artist_counts"],
        app_data["playlist_sizes"],
        top_n=top_n,
    )


@app.get("/api/top-tracks")
def get_top_tracks(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    tc = app_data["track_counts"].head(limit).reset_index(drop=True)
    tc["rank"] = tc.index + 1
    return tc[["rank", "track_name", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/top-artists")
def get_top_artists(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    ac = app_data["artist_counts"].head(limit).reset_index(drop=True)
    ac["rank"] = ac.index + 1
    return ac[["rank", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/tracks")
def get_tracks(
    page: int = Query(0, ge=0),
    size: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    tracks = app_data["tracks"]
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
    tc = app_data["track_counts"]
    matches = tc[tc["track_name"].str.contains(q, case=False, na=False, regex=False)]
    return matches.head(limit)[["track_uri", "track_name", "artist_name", "count"]].to_dict(orient="records")


@app.get("/api/recommend/popular")
def recommend(
    seed_tracks: list[str] = Query(..., description="List of track URIs/Spotify URI"),
    limit: int = Query(20, ge=1, le=200),
) -> list[RecommendationScore]:
    return PopularityRecommender(app_data["track_counts"]).recommend(seed_tracks, limit=limit)

@app.get("/api/recommend/cooccurence")
def recommend_cooccurence(
    seed_tracks: list[str] = Query(..., description="List of track URIs/Spotify URI"),
    limit: int = Query(20, ge=1, le=200),
) -> list[RecommendationScore]:
    return ItemItemRecommender(
        app_data["interaction_matrix"],
        app_data["id_mappings"],
        app_data["track_counts"],
    ).recommend(seed_tracks, limit=limit)