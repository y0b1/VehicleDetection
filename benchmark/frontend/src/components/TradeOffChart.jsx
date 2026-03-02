import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Label,
} from 'recharts'

const CONFIGS = ['YOLOv8', 'YOLOv9', 'NMS Ensemble', 'WBF Ensemble']
const COLORS = {
  'YOLOv8': '#6366f1',
  'YOLOv9': '#f59e0b',
  'NMS Ensemble': '#22c55e',
  'WBF Ensemble': '#ec4899',
}

const CustomDot = ({ cx, cy, payload }) => (
  <g>
    <circle
      cx={cx}
      cy={cy}
      r={8}
      fill={COLORS[payload.name]}
      fillOpacity={0.9}
      stroke={COLORS[payload.name]}
      strokeWidth={2}
    />
    <text
      x={cx}
      y={cy - 14}
      textAnchor="middle"
      fill={COLORS[payload.name]}
      fontSize={11}
      fontWeight="600"
    >
      {payload.name.replace(' Ensemble', '')}
    </text>
  </g>
)

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div style={{
      background: '#111111',
      border: '1px solid #333333',
      borderRadius: 8,
      padding: '8px 14px',
    }}>
      <p style={{ color: COLORS[p.name], fontWeight: 600, marginBottom: 4 }}>{p.name}</p>
      <p style={{ color: '#f4f4f5', margin: '2px 0' }}>FPS: {p.x.toFixed(1)}</p>
      <p style={{ color: '#f4f4f5', margin: '2px 0' }}>mAP@0.5: {(p.y * 100).toFixed(1)}%</p>
    </div>
  )
}

export default function TradeOffChart({ results }) {
  const allFps = CONFIGS.map(c => results[c]?.fps ?? 0)
  const midFps = (Math.min(...allFps) + Math.max(...allFps)) / 2

  const allMAP = CONFIGS.map(c => results[c]?.mAP50 ?? 0)
  const midMAP = (Math.min(...allMAP) + Math.max(...allMAP)) / 2

  const data = CONFIGS.map(config => ({
    x: results[config]?.fps ?? 0,
    y: results[config]?.mAP50 ?? 0,
    name: config,
  }))

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-xl p-5">
      <h2 className="text-[#f4f4f5] font-semibold text-lg mb-1">Speed vs Accuracy Trade-off</h2>
      <p className="text-[#71717a] text-xs mb-4">Upper-right = best. FPS (x) vs mAP@0.5 (y).</p>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222222" />
          <XAxis
            dataKey="x"
            type="number"
            name="FPS"
            tick={{ fill: '#71717a', fontSize: 11 }}
            axisLine={{ stroke: '#333333' }}
            tickLine={false}
          >
            <Label value="FPS" position="insideBottom" offset={-15} fill="#71717a" fontSize={12} />
          </XAxis>
          <YAxis
            dataKey="y"
            type="number"
            name="mAP@0.5"
            domain={[0, 1]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: '#71717a', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          >
            <Label value="mAP@0.5" angle={-90} position="insideLeft" offset={15} fill="#71717a" fontSize={12} />
          </YAxis>
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            x={midFps}
            stroke="#333333"
            strokeDasharray="4 4"
          />
          <ReferenceLine
            y={midMAP}
            stroke="#333333"
            strokeDasharray="4 4"
          />
          <Scatter data={data} shape={<CustomDot />} />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-4 mt-1 justify-center">
        {CONFIGS.map(c => (
          <div key={c} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[c] }} />
            <span style={{ color: '#71717a', fontSize: 12 }}>{c}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
