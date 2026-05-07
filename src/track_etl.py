from typing import Union, List, Iterator, Dict, Any
import os
import json
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, hash, explode
from src.utils.schema import get_track_schema
from src.utils.logging import get_logger
from src.utils.spark import spark_session
from scipy.sparse import coo_matrix, save_npz
import numpy as np
import pandas as pd

# Logger for tracking ETL progress and errors
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
            }


def transform_track(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip Spotify URI prefixes from track_uri and album_uri in a single record dict."""
    result = dict(record)
    for key, prefix in (("track_uri", "spotify:track:"), ("album_uri", "spotify:album:")):
        val = result.get(key, "")
        if val.startswith(prefix):
            result[key] = val[len(prefix):]
    return result


def load_tracks(file_path: str, output_dir: str) -> int:
    """
    Read a single JSON slice, transform each track, and write a Parquet file.

    Returns the number of tracks written (0 if the slice had no tracks).
    """
    rows = [transform_track(r) for r in extract_tracks(file_path)]
    if not rows:
        return 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    out_path = Path(output_dir) / f"{stem}.parquet"
    pd.DataFrame(rows).to_parquet(str(out_path), index=False)
    return len(rows)


def spark_extract_tracks(spark, input_path: Union[str, List[str]]) -> DataFrame:
    """
    Extract raw track data from JSON files into a Spark DataFrame.

    Args:
        spark: SparkSession instance.
        input_path: Path or list of paths to input JSON files.

    Returns:
        DataFrame: Raw track data with schema applied.
    """
    logger.info(f"Reading input: {input_path}")

    return (
        spark.read
        .option("multiLine", True)
        .schema(get_track_schema())
        .json(input_path)
    )


def transform_tracks(df: DataFrame) -> DataFrame:
    """
    Flatten playlists and tracks, clean URIs, and encode track_uri to numeric IDs.

    Args:
        df: Raw DataFrame with nested playlists and tracks.

    Returns:
        DataFrame: Flattened, cleaned, and ML-ready track DataFrame with numeric track_id.
    """
    # Flatten the tracks
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
        regexp_replace(col("track.artist_uri"), "spotify:artist:", "").alias("artist_uri")
    )

    # Encode track_uri, album_uri, and artist_uri to numeric IDs using hash function
    df_encoded = df_flat \
        .withColumn("track_id_int", hash(col("track_uri"))) \
        .withColumn("album_id_int", hash(col("album_uri"))) \
        .withColumn("artist_id_int", hash(col("artist_uri")))


    return df_encoded


def spark_load_tracks(df: DataFrame, output_path: str, partitions: int = 8) -> None:
    """Write the transformed Spark DataFrame to Parquet files."""
    logger.info(f"Writing output to: {output_path}")
    df.coalesce(partitions) \
      .write \
      .mode("overwrite")  \
      .parquet(output_path)

def build_playlist_track_matrix(df: DataFrame, output_path: str) -> None:
    """
    Build a playlist-track interaction matrix for collaborative filtering.
    Saves interaction pairs as Parquet first, then builds sparse COO matrix.

    Args:
        df: Transformed track DataFrame with playlist_id and track_id_int.
        output_path: Directory path to save the interaction matrix.
    """
    logger.info("Building playlist-song interaction matrix...")

    # Step 1: Save interaction pairs as Parquet (distributed, no OOM)
    matrix_pairs_path = os.path.join(output_path, "interaction_pairs")
    df.select(
        col("playlist_id"),
        col("track_id_int")
    ).distinct() \
     .coalesce(8) \
     .write \
     .mode("overwrite") \
     .parquet(matrix_pairs_path)

    logger.info("Saved interaction pairs to Parquet, building sparse matrix...")

    # Step 2: Load back in chunks and build sparse matrix
    pdf = pd.read_parquet(matrix_pairs_path)

    # Remap to contiguous indices
    playlist_idx = pdf["playlist_id"].astype("category").cat.codes.values
    track_idx = pdf["track_id_int"].astype("category").cat.codes.values
    interactions = np.ones(len(pdf), dtype=np.float32)

    # Build sparse COO matrix
    matrix = coo_matrix(
        (interactions, (playlist_idx, track_idx)),
        shape=(playlist_idx.max() + 1, track_idx.max() + 1)
    )

    # Save sparse matrix
    matrix_path = os.path.join(output_path, "interaction_matrix.npz")
    save_npz(matrix_path, matrix)
    logger.info(f"Saved sparse matrix {matrix.shape} to {matrix_path}")


def run_full_etl(input_path: Union[str, List[str]], output_path: str, num_records: int = None) -> None:
    """
    Orchestrate the full ETL process: extract, transform, and load.

    Args:
        input_path: Path(s) to raw JSON files.
        output_path: Path to store cleaned Parquet files.
        num_records: If set, limit the number of raw records processed.
    """
    # Initialize Spark session
    spark = spark_session()
    logger.info("Starting ETL job")

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Step 1: Extract
    df = spark_extract_tracks(spark, input_path)
    if num_records:
        logger.info(f"Limiting to {num_records} records")
        df = df.limit(num_records)

    # Step 2: Transform
    df_transformed = transform_tracks(df)

    # Step 3: Load
    spark_load_tracks(df_transformed, output_path)

    logger.info("ETL pipeline finished")
    
    build_playlist_track_matrix(df_transformed, output_path)
    logger.info("Completed building playlist-track interaction matrix")
    
    # Stop the Spark session to release resources
    spark.stop()