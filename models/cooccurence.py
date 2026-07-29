import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from utils.normalize_uri import normalize_uri
from utils.pydantic_schemas import RecommendationScore

class ItemItemRecommender:
    """
    If song X is in playlist, Y would be more likely to be in the same playlist.
    """
    def __init__(self, interaction_matrix: csr_matrix, id_mappings: pd.DataFrame, track_counts: pd.DataFrame):
        self.interaction_matrix = interaction_matrix.tocsr()
        self.id_mappings = id_mappings
        self.track_counts = track_counts.set_index("track_uri")

        # index mappings
        self.uri_to_id = id_mappings.set_index("track_uri")["track_id"]
        self.id_to_uri = id_mappings.set_index("track_id")["track_uri"]

    def recommend(self, seed_tracks: list[str], limit: int = 20) -> list[RecommendationScore]:
        seen = {normalize_uri(uri) for uri in seed_tracks}
        seed_ids = [int(i) for i in self.uri_to_id.reindex(list(seen)).dropna().values]
        if not seed_ids:
            return []
        
        seed_cols = self.interaction_matrix[:,seed_ids] # n playlists x n seeds
        co_occurence = self.interaction_matrix.T @ seed_cols # n tracks x n seeds
        counts = np.asarray(co_occurence.sum(axis=1)).ravel() # n tracks having counts

        counts[seed_ids] = 0
        if not np.any(counts):
            return []

        top_ids = np.argsort(counts)[::-1][:limit] # find top ids
        top_ids = top_ids[counts[top_ids] > 0]

        # build the result
        results = []
        for track_id in top_ids:
            count = int(counts[track_id])
            if count <= 0:
                continue
            uri = self.id_to_uri.get(track_id)
            if uri is None or uri not in self.track_counts.index:
                continue
            row = self.track_counts.loc[uri]
            results.append(RecommendationScore(
                track_uri=uri,
                track_name=row.track_name,
                artist_name=row.artist_name,
                count=count,
                score=float(count),
            ))
        return results