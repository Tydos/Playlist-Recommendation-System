export default function TopTracks({ tracks }) {
  return (
    <div className="panel">
      <h2>Top Tracks</h2>
      <table className="track-table">
        <thead>
          <tr>
            <th className="rank">#</th>
            <th>Track</th>
            <th>Artist</th>
            <th>Playlists</th>
          </tr>
        </thead>
        <tbody>
          {tracks.map((t, i) => (
            <tr key={t.track_name + t.artist_name + i}>
              <td className="rank">{i + 1}</td>
              <td title={t.track_name}>{t.track_name}</td>
              <td title={t.artist_name}>{t.artist_name}</td>
              <td><span className="count-badge">{t.count.toLocaleString()}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
