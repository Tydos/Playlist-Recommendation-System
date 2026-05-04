"""
Train a KNN model on the playlist-track interaction matrix and save artifacts.
"""
import time
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.knn_utils import train_knn
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_knn")


def run_knn_training() -> float:
    config = load_config()
    processed_path = config.get("processed_data_path", "processed_data")
    matrix_path = f"{processed_path}/interaction_matrix.npz"
    pairs_path = f"{processed_path}/interaction_pairs"

    logger.info(f"Loading interaction matrix from {matrix_path}")
    matrix = load_npz(matrix_path).tocsc()
    logger.info(f"Matrix shape: {matrix.shape} (playlists × tracks)")

    logger.info("Loading track index mappings from interaction pairs")
    pdf = pd.read_parquet(pairs_path)
    track_categories = pdf["track_id_int"].astype("category").cat
    idx_to_track = dict(enumerate(track_categories.categories))
    track_to_idx = {v: k for k, v in idx_to_track.items()}
    logger.info(f"Indexed {len(track_to_idx)} unique tracks")

    # Tracks are columns in the matrix; transpose so each track is a row.
    X_items = matrix.T
    logger.info(f"Training KNN on item matrix of shape {X_items.shape}")

    t0 = time.perf_counter()
    knn_model = train_knn(X_items, n_neighbors=10, metric="cosine")
    duration = time.perf_counter() - t0

    artifacts = {
        "knn_model": knn_model,
        "track_to_idx": track_to_idx,
        "idx_to_track": idx_to_track,
    }
    with open("knn_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("Saved knn_artifacts.pkl")

    return duration


if __name__ == "__main__":
    duration = run_knn_training()
    logger.info(f"KNN training completed in {duration:.2f} seconds")
