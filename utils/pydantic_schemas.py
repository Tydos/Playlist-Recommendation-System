from pydantic import BaseModel

class RecommendationScore(BaseModel):
    track_uri: str
    track_name: str
    artist_name: str
    count: int
    score: float
