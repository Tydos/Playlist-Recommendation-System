from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, _data

_TRACK_COUNTS = pd.DataFrame([
    {"track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "count": 10},
    {"track_uri": "def", "track_name": "Song B", "artist_name": "Artist B", "count": 5},
])
_ARTIST_COUNTS = pd.DataFrame([
    {"artist_name": "Artist A", "count": 10},
    {"artist_name": "Artist B", "count": 5},
])
_PLAYLIST_SIZES = pd.DataFrame([
    {"playlist_id": 0, "track_count": 4},
    {"playlist_id": 1, "track_count": 6},
])
_TRACKS = pd.DataFrame([
    {"playlist_id": 0, "track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "album_name": "Album A"},
])

_PARQUET_MAP = {
    "track_counts": _TRACK_COUNTS,
    "artist_counts": _ARTIST_COUNTS,
    "playlist_sizes": _PLAYLIST_SIZES,
    "tracks": _TRACKS,
}


def _make_load_parquet(mapping):
    def _load(path):
        for key, df in mapping.items():
            if key in str(path):
                return df.copy()
        raise FileNotFoundError(path)
    return _load


@pytest.fixture
def loaded_client():
    with patch("api.main._load_parquet", side_effect=_make_load_parquet(_PARQUET_MAP)):
        with TestClient(app) as client:
            yield client


def test_startup_fails_when_etl_not_run():
    _data.clear()
    with patch("api.main._load_parquet", side_effect=FileNotFoundError("no data")):
        with pytest.raises(FileNotFoundError):
            with TestClient(app):
                pass
    _data.clear()


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

def test_stats_returns_correct_shape(loaded_client):
    r = loaded_client.get("/api/stats?top_n=2")
    assert r.status_code == 200
    body = r.json()
    assert body["total_tracks"] == 2
    assert body["total_artists"] == 2
    assert body["total_playlists"] == 2
    assert body["avg_playlist_size"] == 5.0
    assert len(body["top_tracks"]) == 2
    assert len(body["top_artists"]) == 2


def test_stats_top_n_is_respected(loaded_client):
    r = loaded_client.get("/api/stats?top_n=1")
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_tracks"]) == 1
    assert len(body["top_artists"]) == 1


# ---------------------------------------------------------------------------
# /api/top-tracks
# ---------------------------------------------------------------------------

def test_top_tracks_returns_ranked_list(loaded_client):
    r = loaded_client.get("/api/top-tracks?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["rank"] == 1
    assert body[0]["track_name"] == "Song A"
    assert body[0]["count"] == 10


# ---------------------------------------------------------------------------
# /api/top-artists
# ---------------------------------------------------------------------------

def test_top_artists_returns_ranked_list(loaded_client):
    r = loaded_client.get("/api/top-artists?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["rank"] == 1
    assert body[0]["artist_name"] == "Artist A"


# ---------------------------------------------------------------------------
# /api/recommend
# ---------------------------------------------------------------------------

def test_recommend_excludes_seed_tracks(loaded_client):
    r = loaded_client.get("/api/recommend", params={"seed_tracks": ["abc"], "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert all(item["track_uri"] != "abc" for item in body)
    assert body[0]["track_name"] == "Song B"


def test_recommend_normalizes_spotify_uri(loaded_client):
    r = loaded_client.get(
        "/api/recommend",
        params={"seed_tracks": ["spotify:track:abc"], "limit": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert all(item["track_uri"] != "abc" for item in body)


# ---------------------------------------------------------------------------
# /api/tracks/search
# ---------------------------------------------------------------------------

def test_search_tracks_matches_by_name_case_insensitive(loaded_client):
    r = loaded_client.get("/api/tracks/search", params={"q": "song a"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["track_uri"] == "abc"
    assert body[0]["track_name"] == "Song A"


def test_search_tracks_respects_limit(loaded_client):
    r = loaded_client.get("/api/tracks/search", params={"q": "song", "limit": 1})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_search_tracks_returns_empty_for_no_match(loaded_client):
    r = loaded_client.get("/api/tracks/search", params={"q": "nonexistent"})
    assert r.status_code == 200
    assert r.json() == []
