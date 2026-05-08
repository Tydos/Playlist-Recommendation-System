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

const SOURCE_LABELS = {
  api:      { label: 'Live API',    className: 'live' },
  fallback: { label: 'Sample Data', className: 'fallback' },
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

  const badge = source ? SOURCE_LABELS[source] : null

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Playlist Dashboard</h1>
          <p>Spotify Million Playlist Dataset: track and artist analytics</p>
        </div>

        {badge && (
          <div className={`source-badge ${badge.className}`}>
            <span className="dot" />
            {badge.label}
          </div>
        )}
      </header>

      {source === 'fallback' && (
        <div className="warning-banner">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          Live data unavailable; showing sample data. Run the ETL and start the backend to see real results.
        </div>
      )}

      {!stats && (
        <div className="loading-wrap">
          <div className="loading-spinner" />
          <p className="loading-text">Loading pipeline output...</p>
        </div>
      )}

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
              accent="purple"
            />
            <StatCard
              label="Playlists"
              value={stats.total_playlists.toLocaleString()}
              sub="in dataset"
              accent="blue"
            />
            <StatCard
              label="Avg Playlist Size"
              value={stats.avg_playlist_size}
              sub="tracks per playlist"
              accent="orange"
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
