from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app, _data

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
    with patch("backend.api.main._load_parquet", side_effect=_make_load_parquet(_PARQUET_MAP)):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def empty_client():
    _data.clear()
    with patch("backend.api.main._load_parquet", side_effect=FileNotFoundError("no data")):
        with TestClient(app) as client:
            yield client
        _data.clear()


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

def test_stats_returns_503_when_etl_not_run(empty_client):
    r = empty_client.get("/api/stats")
    assert r.status_code == 503


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

def test_top_tracks_returns_503_when_etl_not_run(empty_client):
    r = empty_client.get("/api/top-tracks")
    assert r.status_code == 503


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

def test_top_artists_returns_503_when_etl_not_run(empty_client):
    r = empty_client.get("/api/top-artists")
    assert r.status_code == 503


def test_top_artists_returns_ranked_list(loaded_client):
    r = loaded_client.get("/api/top-artists?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["rank"] == 1
    assert body[0]["artist_name"] == "Artist A"
