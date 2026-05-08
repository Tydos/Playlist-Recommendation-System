function rankClass(i) {
  if (i === 0) return 'gold'
  if (i === 1) return 'silver'
  if (i === 2) return 'bronze'
  return ''
}

export default function TopTracks({ tracks }) {
  if (!tracks?.length) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Top Tracks</span>
        </div>
        <p className="panel-empty">No track data available.</p>
      </div>
    )
  }

  const max = tracks[0].count

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Top Tracks</span>
        <span className="panel-count">{tracks.length} tracks</span>
      </div>

      <ol className="track-list" aria-label="Top tracks by playlist appearances">
        {tracks.map((t, i) => {
          const rc = rankClass(i)
          return (
            <li
              className="track-item"
              key={`${t.track_name}-${t.artist_name}-${i}`}
              style={{
                '--bar-pct': t.count / max,
                '--bar-delay': `${i * 45}ms`,
              }}
            >
              <div className="track-bar" aria-hidden="true" />
              <span
                className={rc ? `track-rank ${rc}` : 'track-rank'}
                aria-label={`Rank ${i + 1}`}
              >
                {i + 1}
              </span>
              <div className="track-info">
                <div className="track-name" title={t.track_name}>{t.track_name}</div>
                <div className="track-artist" title={t.artist_name}>{t.artist_name}</div>
              </div>
              <span className="count-badge">{t.count.toLocaleString()}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
