import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const GREEN = '#1db954'
const DIM = '#d0d0e8'

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { artist_name, count } = payload[0].payload
  return (
    <div style={{ background: '#ffffff', border: '1px solid #e4e4f0', borderRadius: 6, padding: '8px 12px', fontSize: '0.82rem' }}>
      <div style={{ color: '#555' }}>{artist_name}</div>
      <div style={{ color: GREEN, fontWeight: 600 }}>{count.toLocaleString()} playlists</div>
    </div>
  )
}

export default function TopArtists({ artists }) {
  const max = artists[0]?.count ?? 1
  return (
    <div className="panel">
      <h2>Top Artists</h2>
      <ResponsiveContainer width="100%" height={artists.length * 32 + 16}>
        <BarChart
          data={artists}
          layout="vertical"
          margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
          barSize={14}
        >
          <XAxis type="number" hide domain={[0, max]} />
          <YAxis
            type="category"
            dataKey="artist_name"
            width={120}
            tick={{ fill: '#888', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f0f0f8' }} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {artists.map((_, i) => (
              <Cell key={i} fill={i === 0 ? GREEN : DIM} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
