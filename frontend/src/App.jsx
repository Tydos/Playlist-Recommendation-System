import { useEffect, useState } from 'react'
import './App.css'
import StatCard from './components/StatCard'
import TopTracks from './components/TopTracks'
import TopArtists from './components/TopArtists'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/stats?top_n=10`)
      .then(r => {
        if (!r.ok) throw new Error(`API returned ${r.status}`)
        return r.json()
      })
      .then(setStats)
      .catch(e => setError(e.message))
  }, [])

  return (
    <div className="app">
      <div className="header">
        <h1>Collaborative Playlist Dashboard</h1>
        <p>Spotify Million Playlist Dataset — track &amp; artist analytics</p>
      </div>

      {error && (
        <div className="error-box">
          Could not reach API: {error}. Start the backend with{' '}
          <code>uvicorn backend.api.main:app --reload</code>.
        </div>
      )}

      {!stats && !error && <div className="loading">Loading data…</div>}

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
