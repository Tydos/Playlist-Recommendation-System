const ACCENTS = {
  green:  { bg: 'oklch(96%   0.018 265)', border: 'oklch(87%   0.032 265)', value: 'oklch(33% 0.092 265)' },
  purple: { bg: 'oklch(94.5% 0.024 265)', border: 'oklch(85%   0.042 265)', value: 'oklch(29% 0.078 265)' },
  blue:   { bg: 'oklch(95.5% 0.020 265)', border: 'oklch(86%   0.036 265)', value: 'oklch(37% 0.085 265)' },
  orange: { bg: 'oklch(96.5% 0.014 265)', border: 'oklch(88%   0.024 265)', value: 'oklch(42% 0.065 265)' },
}

export default function StatCard({ label, value, sub, accent = 'green' }) {
  const { bg, border, value: valueColor } = ACCENTS[accent] ?? ACCENTS.green

  return (
    <div
      className="stat-card"
      style={{ '--card-bg': bg, '--card-border': border, '--value-color': valueColor }}
    >
      <div className="value">{value}</div>
      <div className="label">{label}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}
