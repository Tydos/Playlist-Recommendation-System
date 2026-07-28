"""
Baseline popularity model - recommend the most popular tracks in the dataset.
"""

import pandas as pd

class PopularityRecommender:
    def __init__(self, track_counts: pd.DataFrame):
        self.track_counts = track_counts

    @staticmethod
    def _normalize_uri(uri: str) -> str:
        return uri.removeprefix("spotify:track:")

    def recommend(self, seed_uris: list[str], limit: int = 20) -> list[dict]:
        seen = {self._normalize_uri(u) for u in seed_uris}
        hits = self.track_counts[~self.track_counts["track_uri"].isin(seen)]
        return (
            hits.head(limit)
            .assign(score=lambda df: df["count"])
            .to_dict(orient="records")
        )