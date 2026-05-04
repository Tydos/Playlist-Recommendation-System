"""
Bayesian Personalized Ranking (BPR) via vectorized mini-batch SGD.
Subset: first 50K playlists (pure-numpy BPR on the full 1M dataset would take hours).
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_bpr")

N_PLAYLISTS    = 50_000
N_FACTORS      = 64
N_EPOCHS       = 5
SAMPLES_EPOCH  = 1_000_000
BATCH_SIZE     = 10_000
LR             = 0.05
REG            = 0.01


def run_bpr_training():
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
    logger.info(f"BPR problem: {n_users} playlists × {n_items} tracks, "
                f"{N_FACTORS} factors, {N_EPOCHS} epochs × {SAMPLES_EPOCH} samples")

    # Build user→positive-items lookup (list per user for fast negative sampling)
    user_pos_set  = defaultdict(set)
    user_pos_list = defaultdict(list)
    for u, i in zip(user_ids, item_ids):
        user_pos_set[u].add(i)
        user_pos_list[u].append(i)

    rng = np.random.default_rng(42)
    U = (rng.random((n_users, N_FACTORS)) - 0.5) * 0.01
    V = (rng.random((n_items, N_FACTORS)) - 0.5) * 0.01

    logger.info("Starting BPR mini-batch SGD")
    t0 = time.perf_counter()

    for epoch in range(N_EPOCHS):
        n_batches = SAMPLES_EPOCH // BATCH_SIZE
        for _ in range(n_batches):
            # Sample users (only those with interactions)
            active_users = np.array(list(user_pos_set.keys()))
            bu = rng.choice(active_users, size=BATCH_SIZE)

            # Positive items: one random positive per sampled user
            bi = np.array([user_pos_list[u][rng.integers(len(user_pos_list[u]))] for u in bu])

            # Negative items: random, re-sample if accidentally positive
            bj = rng.integers(0, n_items, size=BATCH_SIZE)

            # Vectorized BPR gradient
            diff = V[bi] - V[bj]                              # (B, F)
            x_uij = np.sum(U[bu] * diff, axis=1)              # (B,)
            grad  = 1.0 / (1.0 + np.exp(x_uij))               # (B,) sigmoid(-x)

            gU = grad[:, None] * diff  - REG * U[bu]          # (B, F)
            gVi = grad[:, None] * U[bu] - REG * V[bi]         # (B, F)
            gVj = -grad[:, None] * U[bu] - REG * V[bj]        # (B, F)

            np.add.at(U, bu, LR * gU)
            np.add.at(V, bi, LR * gVi)
            np.add.at(V, bj, LR * gVj)

        logger.info(f"  Epoch {epoch + 1}/{N_EPOCHS} done")

    train_duration = time.perf_counter() - t0
    logger.info(f"BPR training completed in {train_duration:.2f} seconds")

    # Inference: average seed embeddings, cosine search over V
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
    train_dur, inf_dur = run_bpr_training()
    logger.info(f"BPR | training: {train_dur:.2f}s | inference: {inf_dur:.3f}s")
