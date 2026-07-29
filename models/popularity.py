import pandas as pd
from utils.pydantic_schemas import RecommendationScore 
from utils.normalize_uri import normalize_uri

class PopularityRecommender:
    """
    Recommend the most popular tracks in the dataset excluding seed tracks.
    """
    def __init__(self, track_counts: pd.DataFrame):
        self.track_counts = track_counts # track counts is a df with columns: track_uri, count, artist_name, track_name

    def recommend(self, seed_uris: list[str], limit: int = 20) -> list[RecommendationScore]:
        """Do not return tracks that are in the seed list."""
        seen = {normalize_uri(uri) for uri in seed_uris}

        popular_tracks = self.track_counts.head(limit) # top 20 most popular tracks
        popular_tracks_without_seeds = popular_tracks[~popular_tracks["track_uri"].isin(seen)] # exclude seed tracks

        return [
            RecommendationScore (
                track_uri=row.track_uri,
                count=row.count,
                score=row.count,
                artist_name=row.artist_name,
                track_name=row.track_name,
            )
            for row in popular_tracks_without_seeds.itertuples(index=False)
        ]