import {
  BarChart as RechartsBar,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const CONFIGS = ['YOLOv8', 'RT-DETR', 'NMS Ensemble', 'WBF Ensemble']
const COLORS = {
  'YOLOv8': '#6366f1',
  'RT-DETR': '#f59e0b',
  'NMS Ensemble': '#22c55e',
  'WBF Ensemble': '#ec4899',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#111111',
      border: '1px solid #333333',
      borderRadius: 8,
      padding: '8px 14px',
    }}>
      <p style={{ color: '#f4f4f5', fontWeight: 600, marginBottom: 4 }}>{label}</p>
      <p style={{ color: '#6366f1', margin: 0 }}>
        mAP@0.5: {(payload[0].value * 100).toFixed(1)}%
      </p>
    </div>
  )
}

export default function BarChart({ results }) {
  const data = CONFIGS.map(config => ({
    name: config,
    mAP50: results[config]?.mAP50 ?? 0,
  }))

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-xl p-5">
      <h2 className="text-[#f4f4f5] font-semibold text-lg mb-4">mAP@0.5 Comparison</h2>
      <ResponsiveContainer width="100%" height={260}>
        <RechartsBar data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222222" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: '#71717a', fontSize: 11 }}
            axisLine={{ stroke: '#333333' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: '#71717a', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(99,102,241,0.08)' }} />
          <Bar dataKey="mAP50" radius={[4, 4, 0, 0]} maxBarSize={56}>
            {data.map(entry => (
              <Cell key={entry.name} fill={COLORS[entry.name]} />
            ))}
          </Bar>
        </RechartsBar>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-4 mt-3 justify-center">
        {CONFIGS.map(c => (
          <div key={c} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: COLORS[c] }} />
            <span style={{ color: '#71717a', fontSize: 12 }}>{c}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
