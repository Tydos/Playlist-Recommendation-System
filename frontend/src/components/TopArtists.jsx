function rankClass(i) {
  if (i === 0) return 'gold'
  if (i === 1) return 'silver'
  if (i === 2) return 'bronze'
  return ''
}

export default function TopArtists({ artists }) {
  if (!artists?.length) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Top Artists</span>
        </div>
        <p className="panel-empty">No artist data available.</p>
      </div>
    )
  }

  const max = artists[0].count

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Top Artists</span>
        <span className="panel-count">{artists.length} artists</span>
      </div>

      <ol className="track-list" aria-label="Top artists by playlist appearances">
        {artists.map((a, i) => {
          const rc = rankClass(i)
          return (
            <li
              className="track-item"
              key={`${a.artist_name}-${i}`}
              style={{
                '--bar-pct': a.count / max,
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
                <div className="track-name" title={a.artist_name}>{a.artist_name}</div>
              </div>
              <span className="count-badge">{a.count.toLocaleString()}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
