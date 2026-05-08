import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scipy.sparse import load_npz

from backend.etl.track_etl import (
    build_id_mappings,
    build_playlist_track_matrix,
    build_stats,
    build_stats_json,
    extract_tracks,
    load_tracks,
    transform_track,
)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _make_tracks_parquet(path: Path) -> pd.DataFrame:
    df = pd.DataFrame([
        {"playlist_id": 0, "track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "album_name": "Album A", "album_uri": "alb1", "artist_uri": "art1"},
        {"playlist_id": 0, "track_uri": "def", "track_name": "Song B", "artist_name": "Artist B", "album_name": "Album B", "album_uri": "alb2", "artist_uri": "art2"},
        {"playlist_id": 1, "track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "album_name": "Album A", "album_uri": "alb1", "artist_uri": "art1"},
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# extract_tracks
# ---------------------------------------------------------------------------

def test_extract_tracks_outputs_expected_fields(tmp_path):
    file_path = tmp_path / "slice.json"
    _write_json(
        file_path,
        {
            "playlists": [
                {
                    "tracks": [
                        {
                            "track_uri": "spotify:track:111",
                            "artist_name": "Artist A",
                            "track_name": "Song A",
                            "album_name": "Album A",
                            "album_uri": "spotify:album:aaa",
                            "artist_uri": "spotify:artist:111a",
                        }
                    ]
                },
                {
                    "tracks": [
                        {
                            "track_uri": "spotify:track:222",
                            "artist_name": "Artist B",
                            "track_name": "Song B",
                            "album_name": "Album B",
                            "album_uri": "spotify:album:bbb",
                            "artist_uri": "spotify:artist:222b",
                        }
                    ]
                },
            ]
        },
    )

    rows = list(extract_tracks(str(file_path)))
    assert len(rows) == 2
    assert rows[0]["playlist_id"] == 0
    assert rows[0]["track_uri"] == "spotify:track:111"
    assert rows[0]["artist_name"] == "Artist A"
    assert rows[0]["track_name"] == "Song A"
    assert rows[0]["album_name"] == "Album A"
    assert rows[0]["album_uri"] == "spotify:album:aaa"
    assert rows[0]["artist_uri"] == "spotify:artist:111a"
    assert rows[1]["playlist_id"] == 1
    assert rows[1]["track_uri"] == "spotify:track:222"


# ---------------------------------------------------------------------------
# transform_track
# ---------------------------------------------------------------------------

def test_transform_track_strips_all_prefixes():
    record = {
        "track_uri": "spotify:track:xyz",
        "album_uri": "spotify:album:abc",
        "artist_uri": "spotify:artist:zzz",
        "track_name": "T",
        "album_name": "A",
        "artist_name": "R",
    }
    result = transform_track(record)
    assert result["track_uri"] == "xyz"
    assert result["album_uri"] == "abc"
    assert result["artist_uri"] == "zzz"


def test_transform_track_handles_empty_uris():
    record = {
        "track_uri": "",
        "album_uri": "",
        "artist_uri": "",
        "track_name": "",
        "album_name": "",
        "artist_name": "",
    }
    result = transform_track(record)
    assert result["track_uri"] == ""
    assert result["album_uri"] == ""
    assert result["artist_uri"] == ""


# ---------------------------------------------------------------------------
# load_tracks
# ---------------------------------------------------------------------------

def test_load_tracks_creates_dir_and_writes_parquet(monkeypatch, tmp_path):
    file_path = tmp_path / "slice.json"
    _write_json(
        file_path,
        {
            "playlists": [
                {
                    "tracks": [
                        {
                            "track_uri": "spotify:track:123",
                            "artist_name": "Artist1",
                            "track_name": "Track1",
                            "album_name": "Album1",
                            "album_uri": "spotify:album:abc",
                            "artist_uri": "spotify:artist:a1",
                        },
                        {
                            "track_uri": "spotify:track:456",
                            "artist_name": "Artist2",
                            "track_name": "Track2",
                            "album_name": "Album2",
                            "album_uri": "spotify:album:def",
                            "artist_uri": "spotify:artist:a2",
                        },
                    ]
                }
            ]
        },
    )

    out_dir = tmp_path / "parquet_out"
    captured = {}

    def fake_to_parquet(self, path, index=False):
        captured["path"] = str(path)
        captured["index"] = index
        captured["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    count = load_tracks(str(file_path), str(out_dir))

    assert count == 2
    assert out_dir.exists()
    assert captured["path"].endswith("slice.parquet")
    assert captured["index"] is False
    assert captured["df"].iloc[0]["track_uri"] == "123"
    assert captured["df"].iloc[0]["album_uri"] == "abc"
    assert captured["df"].iloc[0]["artist_uri"] == "a1"


def test_load_tracks_returns_zero_when_no_tracks(monkeypatch, tmp_path):
    file_path = tmp_path / "empty_slice.json"
    _write_json(file_path, {"playlists": []})

    out_dir = tmp_path / "out"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("to_parquet should not be called when there are no tracks")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_if_called)

    count = load_tracks(str(file_path), str(out_dir))
    assert count == 0


# ---------------------------------------------------------------------------
# build_id_mappings
# ---------------------------------------------------------------------------

def test_build_id_mappings_deduplicates_and_assigns_ids(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("backend.etl.track_etl.upload_file") as mock_upload:
        result = build_id_mappings(str(tracks_path), str(tmp_path))

    assert len(result) == 2  # "abc" and "def" are the two unique track_uris
    assert set(result["track_uri"]) == {"abc", "def"}
    assert set(result["track_id"]) == {0, 1}
    assert (tmp_path / "id_mappings.parquet").exists()
    mock_upload.assert_called_once_with(str(tmp_path / "id_mappings.parquet"), "output/id_mappings.parquet")


# ---------------------------------------------------------------------------
# build_stats
# ---------------------------------------------------------------------------

def test_build_stats_creates_all_three_parquet_files(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("backend.etl.track_etl.upload_dir") as mock_upload:
        build_stats(str(tracks_path), str(tmp_path))

    stats_dir = tmp_path / "stats"
    assert (stats_dir / "track_counts.parquet").exists()
    assert (stats_dir / "artist_counts.parquet").exists()
    assert (stats_dir / "playlist_sizes.parquet").exists()
    mock_upload.assert_called_once_with(str(stats_dir), "output/stats")


def test_build_stats_counts_are_correct(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)  # "abc" appears in 2 playlists, "def" in 1

    with patch("backend.etl.track_etl.upload_dir"):
        build_stats(str(tracks_path), str(tmp_path))

    track_counts = pd.read_parquet(tmp_path / "stats" / "track_counts.parquet")
    assert track_counts.iloc[0]["track_uri"] == "abc"
    assert track_counts.iloc[0]["count"] == 2

    artist_counts = pd.read_parquet(tmp_path / "stats" / "artist_counts.parquet")
    assert artist_counts.iloc[0]["artist_name"] == "Artist A"
    assert artist_counts.iloc[0]["count"] == 2


# ---------------------------------------------------------------------------
# build_stats_json
# ---------------------------------------------------------------------------

def test_build_stats_json_creates_valid_json(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("backend.etl.track_etl.upload_dir"):
        build_stats(str(tracks_path), str(tmp_path))

    with patch("backend.etl.track_etl.upload_file") as mock_upload:
        build_stats_json(str(tracks_path), str(tmp_path), top_n=2)

    stats_file = tmp_path / "stats.json"
    assert stats_file.exists()

    payload = json.loads(stats_file.read_text())
    assert payload["total_tracks"] == 2
    assert payload["total_artists"] == 2
    assert payload["total_playlists"] == 2
    assert payload["avg_playlist_size"] == 1.5
    assert len(payload["top_tracks"]) == 2
    assert len(payload["top_artists"]) == 2
    assert set(payload["top_tracks"][0].keys()) == {"track_name", "artist_name", "count"}
    assert set(payload["top_artists"][0].keys()) == {"artist_name", "count"}
    mock_upload.assert_called_once_with(str(stats_file), "output/stats.json")


def test_build_stats_json_top_n_is_respected(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("backend.etl.track_etl.upload_dir"):
        build_stats(str(tracks_path), str(tmp_path))

    with patch("backend.etl.track_etl.upload_file"):
        build_stats_json(str(tracks_path), str(tmp_path), top_n=1)

    payload = json.loads((tmp_path / "stats.json").read_text())
    assert len(payload["top_tracks"]) == 1
    assert len(payload["top_artists"]) == 1


# ---------------------------------------------------------------------------
# build_playlist_track_matrix
# ---------------------------------------------------------------------------

def test_build_playlist_track_matrix_creates_npz(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("backend.etl.track_etl.upload_file") as mock_upload:
        id_mappings = build_id_mappings(str(tracks_path), str(tmp_path))

    with patch("backend.etl.track_etl.upload_file") as mock_upload:
        build_playlist_track_matrix(str(tracks_path), id_mappings, str(tmp_path))

    npz_path = tmp_path / "interaction_matrix.npz"
    assert npz_path.exists()
    matrix = load_npz(str(npz_path))
    assert matrix.shape[0] == 2   # 2 playlists
    assert matrix.shape[1] == 2   # 2 unique tracks
    assert matrix.nnz == 3        # 3 interactions
    mock_upload.assert_called_once_with(str(npz_path), "output/interaction_matrix.npz")
