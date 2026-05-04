"""
Implicit ALS on the playlist-track interaction matrix.
Subset: first 100K playlists. Gram-matrix trick keeps each per-item/per-user
solve at O(factors^2) rather than O(n_items * factors^2).
"""
import time
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_als")

N_PLAYLISTS = 100_000
N_FACTORS   = 64
N_ITERS     = 3
ALPHA       = 40.0
REG         = 0.1


def _solve_one(gram, pos_vecs, alpha, reg, eye):
    """Solve one ALS weighted-least-squares update."""
    A = gram + alpha * (pos_vecs.T @ pos_vecs) + reg * eye
    b = (1.0 + alpha) * pos_vecs.sum(axis=0)
    return scipy.linalg.solve(A, b, assume_a="pos")


def run_als_training():
    config = load_config()
    processed_path = config.get("processed_data_path", "processed_data")

    logger.info(f"Loading interaction pairs (first {N_PLAYLISTS} playlists)")
    pdf = pd.read_parquet(f"{processed_path}/interaction_pairs")

    top_playlists = np.sort(pdf["playlist_id"].unique())[:N_PLAYLISTS]
    pdf = pdf[pdf["playlist_id"].isin(top_playlists)].copy()

    playlist_cat = pdf["playlist_id"].astype("category").cat
    track_cat    = pdf["track_id_int"].astype("category").cat
    user_ids     = playlist_cat.codes.values.astype(np.int32)
    item_ids     = track_cat.codes.values.astype(np.int32)
    idx_to_track = dict(enumerate(track_cat.categories))
    track_to_idx = {v: k for k, v in idx_to_track.items()}

    n_users = int(user_ids.max()) + 1
    n_items = int(item_ids.max()) + 1
    logger.info(f"ALS problem: {n_users} playlists × {n_items} tracks, {N_FACTORS} factors, {N_ITERS} iters")

    # Build sparse user-item matrix
    data = np.ones(len(user_ids), dtype=np.float32)
    R = csr_matrix((data, (user_ids, item_ids)), shape=(n_users, n_items))
    Rt = R.T.tocsr()

    rng = np.random.default_rng(42)
    U = (rng.random((n_users, N_FACTORS)) - 0.5) * 0.01
    V = (rng.random((n_items, N_FACTORS)) - 0.5) * 0.01
    eye = np.eye(N_FACTORS)

    logger.info("Starting ALS iterations")
    t0 = time.perf_counter()

    for it in range(N_ITERS):
        # --- Update users ---
        VTV = V.T @ V
        for u in range(n_users):
            pos = R[u].indices
            if len(pos) == 0:
                continue
            U[u] = _solve_one(VTV, V[pos], ALPHA, REG, eye)

        # --- Update items ---
        UTU = U.T @ U
        for i in range(n_items):
            pos = Rt[i].indices
            if len(pos) == 0:
                continue
            V[i] = _solve_one(UTU, U[pos], ALPHA, REG, eye)

        logger.info(f"  ALS iteration {it + 1}/{N_ITERS} done")

    train_duration = time.perf_counter() - t0
    logger.info(f"ALS training completed in {train_duration:.2f} seconds")

    # Inference: average seed embeddings, cosine search over V
    sample_playlist = pdf["playlist_id"].iloc[0]
    seed_raw  = pdf[pdf["playlist_id"] == sample_playlist]["track_id_int"].tolist()
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
    train_dur, inf_dur = run_als_training()
    logger.info(f"ALS | training: {train_dur:.2f}s | inference: {inf_dur:.3f}s")
