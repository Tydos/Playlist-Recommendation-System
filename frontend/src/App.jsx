import { useEffect, useState } from 'react'
import './App.css'
import StatCard from './components/StatCard'
import TopTracks from './components/TopTracks'
import TopArtists from './components/TopArtists'
import fallbackStats from './fallbackStats'

const API = import.meta.env.VITE_API_URL ?? ''
const S3_STATS_URL = import.meta.env.VITE_S3_STATS_URL ?? ''

async function fetchStats() {
  if (API) {
    try {
      const r = await fetch(`${API}/api/stats?top_n=10`)
      if (r.ok) return { data: await r.json(), source: 'api' }
    } catch {}
  }

  if (S3_STATS_URL) {
    try {
      const r = await fetch(S3_STATS_URL)
      if (r.ok) return { data: await r.json(), source: 's3' }
    } catch {}
  }

  return { data: fallbackStats, source: 'fallback' }
}

export default function App() {
  const [stats, setStats] = useState(null)
  const [source, setSource] = useState(null)

  useEffect(() => {
    fetchStats().then(({ data, source }) => {
      setStats(data)
      setSource(source)
    })
  }, [])

  return (
    <div className="app">
      <div className="header">
        <h1>Collaborative Playlist Dashboard</h1>
        <p>Spotify Million Playlist Dataset — track &amp; artist analytics</p>
      </div>

      {source === 'fallback' && (
        <div className="error-box">
          Live data unavailable — showing sample data. Run the ETL and start the backend to see real results.
        </div>
      )}

      {!stats && <div className="loading">Loading data…</div>}

      {stats && (
        <>
          <div className="stat-grid">
            <StatCard
              label="Unique Tracks"
              value={stats.total_tracks.toLocaleString()}
              sub="distinct track URIs"
            />
            <StatCard
              label="Artists"
              value={stats.total_artists.toLocaleString()}
              sub="unique artists"
            />
            <StatCard
              label="Playlists"
              value={stats.total_playlists.toLocaleString()}
              sub="in dataset"
            />
            <StatCard
              label="Avg Playlist Size"
              value={stats.avg_playlist_size}
              sub="tracks per playlist"
            />
          </div>

          <div className="charts-row">
            <TopTracks tracks={stats.top_tracks} />
            <TopArtists artists={stats.top_artists} />
          </div>
        </>
      )}
    </div>
  )
}
