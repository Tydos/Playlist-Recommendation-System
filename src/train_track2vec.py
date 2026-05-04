"""
Track2Vec via TruncatedSVD on the full playlist-track interaction matrix.
Randomized SVD on the co-occurrence structure is the mathematical foundation
of Word2Vec SGNS (Levy & Goldberg 2014), making this a valid Track2Vec proxy.
Full dataset: 1M playlists × 2.26M tracks.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_track2vec")

N_COMPONENTS = 64
N_ITER       = 5


def run_track2vec_training():
    config = load_config()
    processed_path = config.get("processed_data_path", "processed_data")
    matrix_path    = f"{processed_path}/interaction_matrix.npz"
    pairs_path     = f"{processed_path}/interaction_pairs"

    logger.info(f"Loading interaction matrix from {matrix_path}")
    matrix = load_npz(matrix_path).tocsr()
    logger.info(f"Matrix shape: {matrix.shape} (playlists × tracks)")

    logger.info("Loading track index mappings")
    pdf = pd.read_parquet(pairs_path)
    track_cat    = pdf["track_id_int"].astype("category").cat
    idx_to_track = dict(enumerate(track_cat.categories))
    track_to_idx = {v: k for k, v in idx_to_track.items()}

    svd = TruncatedSVD(n_components=N_COMPONENTS, algorithm="randomized",
                       n_iter=N_ITER, random_state=42)

    logger.info(f"Fitting TruncatedSVD (n_components={N_COMPONENTS}, n_iter={N_ITER})")
    t0 = time.perf_counter()
    svd.fit(matrix)
    train_duration = time.perf_counter() - t0
    logger.info(f"SVD fit completed in {train_duration:.2f} seconds")

    # svd.components_ shape: (n_components, n_tracks) — rows are latent dims
    # Transpose to get per-track embeddings, then L2-normalise for cosine search
    V = normalize(svd.components_.T)   # (n_tracks, n_components)
    logger.info(f"Track embedding matrix: {V.shape}")

    # Inference: average seed embeddings, cosine search over all tracks
    sample_playlist = pdf["playlist_id"].iloc[0]
    seed_raw    = pdf[pdf["playlist_id"] == sample_playlist]["track_id_int"].tolist()
    seed_tracks = [t for t in seed_raw if t in track_to_idx][:5]
    logger.info(f"Running inference with {len(seed_tracks)} seed tracks")

    t1 = time.perf_counter()
    seed_vecs   = V[[track_to_idx[t] for t in seed_tracks]]
    query_vec   = seed_vecs.mean(axis=0, keepdims=True)
    scores      = cosine_similarity(query_vec, V)[0]
    top_indices = np.argsort(scores)[::-1]
    recs = [idx_to_track[i] for i in top_indices if idx_to_track[i] not in seed_tracks][:10]
    inf_duration = time.perf_counter() - t1

    logger.info(f"Inference: {len(recs)} recommendations in {inf_duration:.3f} seconds")
    return train_duration, inf_duration


if __name__ == "__main__":
    train_dur, inf_dur = run_track2vec_training()
    logger.info(f"Track2Vec(SVD) | training: {train_dur:.2f}s | inference: {inf_dur:.3f}s")
