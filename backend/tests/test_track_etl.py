import json
from pathlib import Path

import pandas as pd

from backend.etl.track_etl import extract_tracks, transform_track, load_tracks


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


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
