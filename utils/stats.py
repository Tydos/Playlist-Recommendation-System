from typing import Any

import pandas as pd


def build_stats(
    track_counts: pd.DataFrame,
    artist_counts: pd.DataFrame,
    playlist_sizes: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    top_tracks = (
        track_counts.head(top_n)[["track_name", "artist_name", "count"]]
        .to_dict(orient="records")
    )
    top_artists = (
        artist_counts.head(top_n)[["artist_name", "count"]]
        .to_dict(orient="records")
    )

    return {
        "total_tracks": len(track_counts),
        "total_artists": len(artist_counts),
        "total_playlists": len(playlist_sizes),
        "avg_playlist_size": round(float(playlist_sizes["track_count"].mean()), 1),
        "top_tracks": top_tracks,
        "top_artists": top_artists,
    }
