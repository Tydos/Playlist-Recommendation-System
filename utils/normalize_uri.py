SPOTIFY_TRACK_PREFIX = "spotify:track:"

def normalize_uri(uri: str) -> str:
    return uri.removeprefix(SPOTIFY_TRACK_PREFIX) # since we stored uri wihtout prefix on track_counts
