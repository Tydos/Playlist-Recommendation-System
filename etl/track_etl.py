import json
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    dense_rank,
    explode,
    hash,
    input_file_name,
    regexp_replace,
)
from pyspark.sql.window import Window
from scipy.sparse import coo_matrix, save_npz

from utils.blob import upload_dir, upload_file
from utils.logging import get_logger
from utils.schema import get_track_schema
from utils.spark import spark_session

logger = get_logger("track_etl")

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
    """Flatten playlists and tracks, clean URIs, and encode track_uri to numeric IDs.
     Assign a (slice_id, pid) pair to each playlist to identify the source slice.
     """
    df_playlists = df.select(
        input_file_name().alias("slice_id"),
        explode("playlists").alias("playlist"),
    ).select(
        col("slice_id"),
        col("playlist.pid").alias("pid"),
        col("playlist.tracks").alias("tracks"),
    )

    unique_playlists = df_playlists.select("slice_id", "pid").distinct()
    window = Window.orderBy("slice_id", "pid")
    unique_playlists = unique_playlists.withColumn("playlist_id", dense_rank().over(window) - 1)
    df_playlists = df_playlists.join(unique_playlists, on=["slice_id", "pid"], how="inner")

    df_flat = df_playlists.select(
        col("slice_id"),
        col("pid"),
        col("playlist_id"),
        explode("tracks").alias("track"),
    ).select(
        col("slice_id"),
        col("pid"),
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
    local_path = Path(output_dir) / "id_mappings.parquet"
    unique_tracks.to_parquet(local_path, index=False)
    logger.info(f"Saved {len(unique_tracks)} unique track mappings")
    upload_file(str(local_path), "output/id_mappings.parquet")
    return unique_tracks

def build_playlist_mappings(tracks_path: str, output_dir: str) -> pd.DataFrame:
    """Build a stable (slice_id, pid) → contiguous playlist_id mapping and save it.
    Returns the mappings DataFrame for downstream use.
    """
    df = pd.read_parquet(tracks_path, columns=["slice_id", "pid", "playlist_id"])
    unique_playlists = (
        df.drop_duplicates(["slice_id", "pid"])
        .sort_values("playlist_id")
        .reset_index(drop=True)
    )
    local_path = Path(output_dir) / "playlist_mappings.parquet"
    unique_playlists.to_parquet(local_path, index=False)
    logger.info(f"Saved {len(unique_playlists)} unique playlist mappings")
    upload_file(str(local_path), "output/playlist_mappings.parquet")
    return unique_playlists


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
    upload_dir(str(stats_dir), "output/stats")


def build_stats_json(tracks_path: str, output_dir: str, top_n: int = 10) -> None:
    """Write a stats.json (matching /api/stats shape) locally and upload to S3."""
    stats_dir = Path(output_dir) / "stats"

    track_counts = pd.read_parquet(stats_dir / "track_counts.parquet")
    artist_counts = pd.read_parquet(stats_dir / "artist_counts.parquet")
    playlist_sizes = pd.read_parquet(stats_dir / "playlist_sizes.parquet")

    payload = {
        "total_tracks": len(track_counts),
        "total_artists": len(artist_counts),
        "total_playlists": len(playlist_sizes),
        "avg_playlist_size": round(float(playlist_sizes["track_count"].mean()), 1),
        "top_tracks": track_counts.head(top_n)[["track_name", "artist_name", "count"]].to_dict(orient="records"),
        "top_artists": artist_counts.head(top_n)[["artist_name", "count"]].to_dict(orient="records"),
    }

    local_path = Path(output_dir) / "stats.json"
    local_path.write_text(json.dumps(payload))
    logger.info(f"Saved stats.json to {local_path}")
    upload_file(str(local_path), "output/stats.json")


def build_playlist_track_matrix(
    tracks_path: str,
    id_mappings: pd.DataFrame,
    playlist_mappings: pd.DataFrame,
    output_dir: str,
) -> None:
    """Build a playlist-track sparse COO matrix using global playlist_id and track_id."""
    logger.info("Building playlist-track interaction matrix...")

    df = pd.read_parquet(tracks_path, columns=["playlist_id", "track_uri"])
    df = df.merge(id_mappings[["track_uri", "track_id"]], on="track_uri", how="inner").drop_duplicates()

    interactions = np.ones(len(df), dtype=np.float32)
    n_playlists = int(playlist_mappings["playlist_id"].max()) + 1
    n_tracks = int(id_mappings["track_id"].max()) + 1

    matrix = coo_matrix(
        (interactions, (df["playlist_id"].values, df["track_id"].values)),
        shape=(n_playlists, n_tracks),
    )

    local_path = Path(output_dir) / "interaction_matrix.npz"
    save_npz(str(local_path), matrix)
    logger.info(f"Saved sparse matrix {matrix.shape} to {output_dir}/interaction_matrix.npz")
    upload_file(str(local_path), "output/interaction_matrix.npz")


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

    upload_dir(tracks_path, "output/tracks.parquet")

    id_mappings = build_id_mappings(tracks_path, output_path)
    playlist_mappings = build_playlist_mappings(tracks_path, output_path)
    build_stats(tracks_path, output_path)
    build_stats_json(tracks_path, output_path)
    build_playlist_track_matrix(tracks_path, id_mappings, playlist_mappings, output_path)

    logger.info("ETL pipeline finished")