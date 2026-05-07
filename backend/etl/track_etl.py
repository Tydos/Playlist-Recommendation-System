from typing import Union, List, Iterator, Dict, Any
import os
import json
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, hash, explode
from backend.utils.schema import get_track_schema
from backend.utils.logging import get_logger
from backend.utils.spark import spark_session
from scipy.sparse import coo_matrix, save_npz
import numpy as np
import pandas as pd

logger = get_logger("track_etl")

def extract_tracks(file_path: str) -> Iterator[Dict[str, Any]]:
    """Yield one dict per track from a single Spotify MPD JSON slice (no Spark)."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    for playlist_id, playlist in enumerate(data.get("playlists", [])):
        for track in playlist.get("tracks", []):
            yield {
                "playlist_id": playlist_id,
                "track_uri": track.get("track_uri", ""),
                "artist_name": track.get("artist_name", ""),
                "track_name": track.get("track_name", ""),
                "album_name": track.get("album_name", ""),
                "album_uri": track.get("album_uri", ""),
                "artist_uri": track.get("artist_uri", ""),
            }


def transform_track(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip Spotify URI prefixes from track_uri, album_uri, and artist_uri."""
    record = dict(record)
    for key, prefix in (
        ("track_uri", "spotify:track:"),
        ("album_uri", "spotify:album:"),
        ("artist_uri", "spotify:artist:"),
    ):
        val = record.get(key, "")
        if val.startswith(prefix):
            record[key] = val[len(prefix):]
    return record


def load_tracks(file_path: str, output_dir: str) -> int:
    """Read a single JSON slice, transform each track, and write a Parquet file.

    Returns the number of tracks written (0 if the slice had no tracks).
    """
    rows = [transform_track(r) for r in extract_tracks(file_path)]
    if not rows:
        return 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = Path(output_dir) / f"{stem}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def spark_extract_tracks(spark, input_path: Union[str, List[str]]) -> DataFrame:
    """Extract raw track data from JSON files into a Spark DataFrame."""
    logger.info(f"Reading input: {input_path}")
    return (
        spark.read
        .option("multiLine", True)
        .schema(get_track_schema())
        .json(input_path)
    )


def transform_tracks(df: DataFrame) -> DataFrame:
    """Flatten playlists and tracks, clean URIs, and encode track_uri to numeric IDs."""
    df_flat = df.select(
        explode("playlists").alias("playlist")
    ).select(
        col("playlist.pid").alias("playlist_id"),
        explode("playlist.tracks").alias("track")
    ).select(
        col("playlist_id"),
        regexp_replace(col("track.track_uri"), "spotify:track:", "").alias("track_uri"),
        col("track.artist_name"),
        col("track.track_name"),
        col("track.album_name"),
        regexp_replace(col("track.album_uri"), "spotify:album:", "").alias("album_uri"),
        regexp_replace(col("track.artist_uri"), "spotify:artist:", "").alias("artist_uri"),
    )

    return df_flat \
        .withColumn("track_id_int", hash(col("track_uri"))) \
        .withColumn("album_id_int", hash(col("album_uri"))) \
        .withColumn("artist_id_int", hash(col("artist_uri")))


def spark_load_tracks(df: DataFrame, output_path: str, partitions: int = 8) -> None:
    """Write the transformed Spark DataFrame to Parquet files."""
    logger.info(f"Writing tracks to: {output_path}")
    df.coalesce(partitions) \
      .write \
      .mode("overwrite") \
      .parquet(output_path)


def build_id_mappings(tracks_path: str, output_dir: str) -> pd.DataFrame:
    """Build a stable track_uri → contiguous int track_id mapping and save it.

    Returns the mappings DataFrame for downstream use.
    """
    df = pd.read_parquet(tracks_path, columns=["track_uri", "track_name", "artist_name", "album_name"])
    unique_tracks = df.drop_duplicates("track_uri").reset_index(drop=True)
    unique_tracks["track_id"] = unique_tracks.index
    unique_tracks.to_parquet(Path(output_dir) / "id_mappings.parquet", index=False)
    logger.info(f"Saved {len(unique_tracks)} unique track mappings")
    return unique_tracks


def build_stats(tracks_path: str, output_dir: str) -> None:
    """Compute and save dashboard aggregations from the tracks parquet."""
    df = pd.read_parquet(tracks_path)
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    track_counts = (
        df.groupby(["track_uri", "track_name", "artist_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    track_counts.to_parquet(stats_dir / "track_counts.parquet", index=False)

    artist_counts = (
        df.groupby("artist_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    artist_counts.to_parquet(stats_dir / "artist_counts.parquet", index=False)

    playlist_sizes = (
        df.groupby("playlist_id")
        .size()
        .reset_index(name="track_count")
        .sort_values("playlist_id")
    )
    playlist_sizes.to_parquet(stats_dir / "playlist_sizes.parquet", index=False)

    logger.info(f"Saved stats: {len(track_counts)} tracks, {len(artist_counts)} artists, {len(playlist_sizes)} playlists")


def build_playlist_track_matrix(tracks_path: str, id_mappings: pd.DataFrame, output_dir: str) -> None:
    """Build a playlist-track sparse COO matrix using contiguous IDs from id_mappings."""
    logger.info("Building playlist-track interaction matrix...")

    df = pd.read_parquet(tracks_path, columns=["playlist_id", "track_uri"])
    df = df.merge(id_mappings[["track_uri", "track_id"]], on="track_uri", how="inner").drop_duplicates()

    playlist_codes = df["playlist_id"].astype("category").cat.codes.values
    track_ids = df["track_id"].values
    interactions = np.ones(len(df), dtype=np.float32)

    n_tracks = int(id_mappings["track_id"].max()) + 1
    matrix = coo_matrix(
        (interactions, (playlist_codes, track_ids)),
        shape=(int(playlist_codes.max()) + 1, n_tracks),
    )

    save_npz(str(Path(output_dir) / "interaction_matrix.npz"), matrix)
    logger.info(f"Saved sparse matrix {matrix.shape} to {output_dir}/interaction_matrix.npz")


def run_full_etl(input_path: Union[str, List[str]], output_path: str, num_records: int = None) -> None:
    """Orchestrate the full ETL process: extract, transform, load, and build outputs."""
    spark = spark_session()
    logger.info("Starting ETL job")
    os.makedirs(output_path, exist_ok=True)

    tracks_path = os.path.join(output_path, "tracks.parquet")

    df = spark_extract_tracks(spark, input_path)
    if num_records:
        logger.info(f"Limiting to {num_records} records")
        df = df.limit(num_records)

    df_transformed = transform_tracks(df)
    spark_load_tracks(df_transformed, tracks_path)
    spark.stop()

    id_mappings = build_id_mappings(tracks_path, output_path)
    build_stats(tracks_path, output_path)
    build_playlist_track_matrix(tracks_path, id_mappings, output_path)

    logger.info("ETL pipeline finished")
