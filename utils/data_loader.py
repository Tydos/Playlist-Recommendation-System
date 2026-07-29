from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import load_npz, spmatrix

from utils.config import load_config

PARQUET_ASSETS = {
    "tracks": "tracks.parquet",
    "track_counts": "stats/track_counts.parquet",
    "artist_counts": "stats/artist_counts.parquet",
    "playlist_sizes": "stats/playlist_sizes.parquet",
    "id_mappings": "id_mappings.parquet",
    "playlist_mappings": "playlist_mappings.parquet",
}

INTERACTION_MATRIX_PATH = "interaction_matrix.npz"

# Keys: tracks, track_counts, artist_counts, playlist_sizes, id_mappings,
# playlist_mappings (pd.DataFrame); interaction_matrix (scipy.sparse.spmatrix)
app_data: dict[str, Any] = {}


def _require_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}. Run the ETL first.")
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(_require_exists(path))


def load_interaction_matrix(path: Path) -> spmatrix:
    return load_npz(_require_exists(path))


def load_etl_output(base_dir: Path | None = None) -> dict[str, Any]:
    """Load all ETL output assets. `base_dir` overrides the config path (used in tests)."""
    out = base_dir or Path(load_config().get("output_path", "output"))

    data = {}
    for key, rel_path in PARQUET_ASSETS.items():
        data[key] = load_parquet(out / rel_path)

    data["interaction_matrix"] = load_interaction_matrix(out / INTERACTION_MATRIX_PATH)

    return data
