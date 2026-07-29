import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scipy.sparse import load_npz

from etl.track_etl import (
    build_id_mappings,
    build_playlist_mappings,
    build_playlist_track_matrix,
    build_stats,
    build_stats_json,
    spark_extract_tracks,
    spark_load_tracks,
    transform_tracks,
)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _track(uri_suffix: str, name: str, artist: str) -> dict:
    return {
        "track_uri": f"spotify:track:{uri_suffix}",
        "artist_name": artist,
        "track_name": name,
        "album_name": f"Album {name}",
        "album_uri": f"spotify:album:alb-{uri_suffix}",
        "artist_uri": f"spotify:artist:art-{uri_suffix}",
    }


def _make_tracks_parquet(path: Path) -> pd.DataFrame:
    df = pd.DataFrame([
        {"slice_id": "slice_a", "pid": 0, "playlist_id": 0, "track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "album_name": "Album A", "album_uri": "alb1", "artist_uri": "art1"},
        {"slice_id": "slice_a", "pid": 0, "playlist_id": 0, "track_uri": "def", "track_name": "Song B", "artist_name": "Artist B", "album_name": "Album B", "album_uri": "alb2", "artist_uri": "art2"},
        {"slice_id": "slice_a", "pid": 1, "playlist_id": 1, "track_uri": "abc", "track_name": "Song A", "artist_name": "Artist A", "album_name": "Album A", "album_uri": "alb1", "artist_uri": "art1"},
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# spark_extract_tracks
# ---------------------------------------------------------------------------

def test_spark_extract_tracks_reads_expected_schema(spark, tmp_path):
    file_path = tmp_path / "slice.json"
    _write_json(
        file_path,
        {
            "playlists": [
                {"pid": 0, "tracks": [_track("111", "Song A", "Artist A")]},
                {"pid": 1, "tracks": [_track("222", "Song B", "Artist B")]},
            ]
        },
    )

    df = spark_extract_tracks(spark, str(file_path))
    row = df.collect()[0]
    playlists = sorted(row["playlists"], key=lambda p: p["pid"])

    assert len(playlists) == 2
    assert playlists[0]["pid"] == 0
    assert playlists[0]["tracks"][0]["track_uri"] == "spotify:track:111"
    assert playlists[1]["pid"] == 1
    assert playlists[1]["tracks"][0]["artist_name"] == "Artist B"


# ---------------------------------------------------------------------------
# transform_tracks
# ---------------------------------------------------------------------------

def test_transform_tracks_flattens_and_strips_uri_prefixes(spark, tmp_path):
    file_path = tmp_path / "slice.json"
    _write_json(
        file_path,
        {"playlists": [{"pid": 0, "tracks": [_track("111", "Song A", "Artist A")]}]},
    )

    raw = spark_extract_tracks(spark, str(file_path))
    result = transform_tracks(raw).toPandas()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["track_uri"] == "111"
    assert row["album_uri"] == "alb-111"
    assert row["artist_uri"] == "art-111"
    assert row["track_name"] == "Song A"
    assert row["artist_name"] == "Artist A"
    assert row["pid"] == 0
    assert row["playlist_id"] == 0


def test_transform_tracks_assigns_unique_playlist_ids_across_slices(spark, tmp_path):
    """Two different slices both use pid=0; playlist_id must disambiguate them globally."""
    file_a = tmp_path / "slice_a.json"
    file_b = tmp_path / "slice_b.json"
    _write_json(file_a, {"playlists": [{"pid": 0, "tracks": [_track("111", "Song A", "Artist A")]}]})
    _write_json(file_b, {"playlists": [{"pid": 0, "tracks": [_track("222", "Song B", "Artist B")]}]})

    raw = spark_extract_tracks(spark, [str(file_a), str(file_b)])
    result = transform_tracks(raw).toPandas()

    assert len(result) == 2
    assert result["pid"].tolist() == [0, 0]
    assert result["slice_id"].nunique() == 2
    assert result["playlist_id"].nunique() == 2


# ---------------------------------------------------------------------------
# spark_load_tracks
# ---------------------------------------------------------------------------

def test_spark_load_tracks_writes_readable_parquet(spark, tmp_path):
    file_path = tmp_path / "slice.json"
    _write_json(
        file_path,
        {"playlists": [{"pid": 0, "tracks": [_track("111", "Song A", "Artist A")]}]},
    )
    out_path = tmp_path / "tracks.parquet"

    raw = spark_extract_tracks(spark, str(file_path))
    transformed = transform_tracks(raw)
    spark_load_tracks(transformed, str(out_path), partitions=1)

    written = pd.read_parquet(out_path)
    assert len(written) == 1
    assert written.iloc[0]["track_uri"] == "111"


# ---------------------------------------------------------------------------
# build_id_mappings
# ---------------------------------------------------------------------------

def test_build_id_mappings_deduplicates_and_assigns_ids(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("etl.track_etl.upload_file") as mock_upload:
        result = build_id_mappings(str(tracks_path), str(tmp_path))

    assert len(result) == 2  # "abc" and "def" are the two unique track_uris
    assert set(result["track_uri"]) == {"abc", "def"}
    assert set(result["track_id"]) == {0, 1}
    assert (tmp_path / "id_mappings.parquet").exists()
    mock_upload.assert_called_once_with(str(tmp_path / "id_mappings.parquet"), "output/id_mappings.parquet")


# ---------------------------------------------------------------------------
# build_playlist_mappings
# ---------------------------------------------------------------------------

def test_build_playlist_mappings_deduplicates_and_sorts(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("etl.track_etl.upload_file") as mock_upload:
        result = build_playlist_mappings(str(tracks_path), str(tmp_path))

    assert len(result) == 2  # playlist_id 0 and 1, deduped from 3 track rows
    assert result["playlist_id"].tolist() == [0, 1]
    assert (tmp_path / "playlist_mappings.parquet").exists()
    mock_upload.assert_called_once_with(str(tmp_path / "playlist_mappings.parquet"), "output/playlist_mappings.parquet")


# ---------------------------------------------------------------------------
# build_stats
# ---------------------------------------------------------------------------

def test_build_stats_creates_all_three_parquet_files(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)

    with patch("etl.track_etl.upload_dir") as mock_upload:
        build_stats(str(tracks_path), str(tmp_path))

    stats_dir = tmp_path / "stats"
    assert (stats_dir / "track_counts.parquet").exists()
    assert (stats_dir / "artist_counts.parquet").exists()
    assert (stats_dir / "playlist_sizes.parquet").exists()
    mock_upload.assert_called_once_with(str(stats_dir), "output/stats")


def test_build_stats_counts_are_correct(tmp_path):
    tracks_path = tmp_path / "tracks.parquet"
    _make_tracks_parquet(tracks_path)  # "abc" appears in 2 playlists, "def" in 1

    with patch("etl.track_etl.upload_dir"):
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

    with patch("etl.track_etl.upload_dir"):
        build_stats(str(tracks_path), str(tmp_path))

    with patch("etl.track_etl.upload_file") as mock_upload:
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

    with patch("etl.track_etl.upload_dir"):
        build_stats(str(tracks_path), str(tmp_path))

    with patch("etl.track_etl.upload_file"):
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

    with patch("etl.track_etl.upload_file"):
        id_mappings = build_id_mappings(str(tracks_path), str(tmp_path))
        playlist_mappings = build_playlist_mappings(str(tracks_path), str(tmp_path))

    with patch("etl.track_etl.upload_file") as mock_upload:
        build_playlist_track_matrix(str(tracks_path), id_mappings, playlist_mappings, str(tmp_path))

    npz_path = tmp_path / "interaction_matrix.npz"
    assert npz_path.exists()
    matrix = load_npz(str(npz_path))
    assert matrix.shape[0] == 2   # 2 playlists
    assert matrix.shape[1] == 2   # 2 unique tracks
    assert matrix.nnz == 3        # 3 interactions
    mock_upload.assert_called_once_with(str(npz_path), "output/interaction_matrix.npz")
