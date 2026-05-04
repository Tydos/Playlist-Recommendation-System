"""
Bayesian Personalized Ranking on the full 1M-playlist interaction matrix using the `implicit` library.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.bpr import BayesianPersonalizedRanking

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_bpr")

N_FACTORS = 64
N_EPOCHS  = 5
LR        = 0.05
REG       = 0.01


def run_bpr_training():
    config = load_config()
    processed_path = config.get("processed_data_path", "processed_data")

    logger.info("Loading full interaction pairs")
    pdf = pd.read_parquet(f"{processed_path}/interaction_pairs")

    playlist_cat = pdf["playlist_id"].astype("category").cat
    track_cat    = pdf["track_id_int"].astype("category").cat
    user_ids     = playlist_cat.codes.values.astype(np.int32)
    item_ids     = track_cat.codes.values.astype(np.int32)
    idx_to_track = dict(enumerate(track_cat.categories))
    track_to_idx = {v: k for k, v in idx_to_track.items()}

    n_users = int(user_ids.max()) + 1
    n_items = int(item_ids.max()) + 1
    logger.info(f"BPR problem: {n_users} playlists × {n_items} tracks, "
                f"{N_FACTORS} factors, {N_EPOCHS} epochs")

    data = np.ones(len(user_ids), dtype=np.float32)
    R = csr_matrix((data, (user_ids, item_ids)), shape=(n_users, n_items))

    model = BayesianPersonalizedRanking(
        factors=N_FACTORS,
        iterations=N_EPOCHS,
        learning_rate=LR,
        regularization=REG,
        use_gpu=False,
        random_state=42,
    )

    logger.info("Starting BPR training")
    t0 = time.perf_counter()
    model.fit(R)
    train_duration = time.perf_counter() - t0
    logger.info(f"BPR training completed in {train_duration:.2f} seconds")

    V = model.item_factors

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
