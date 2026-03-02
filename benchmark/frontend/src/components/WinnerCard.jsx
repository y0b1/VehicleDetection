import { Trophy, Zap, BarChart2, Target } from 'lucide-react'

const CONFIGS = ['YOLOv8', 'RT-DETR', 'NMS Ensemble', 'WBF Ensemble']

export default function WinnerCard({ results }) {
  const bestAccuracy = CONFIGS.reduce((best, c) =>
    (results[c]?.mAP50 ?? 0) > (results[best]?.mAP50 ?? 0) ? c : best
  )
  const bestSpeed = CONFIGS.reduce((best, c) =>
    (results[c]?.fps ?? 0) > (results[best]?.fps ?? 0) ? c : best
  )

  const maxMAP = Math.max(...CONFIGS.map(c => results[c]?.mAP50 ?? 0)) || 1
  const maxFPS = Math.max(...CONFIGS.map(c => results[c]?.fps ?? 0)) || 1
  const bestTradeoff = CONFIGS.reduce((best, c) => {
    const scoreC = 0.6 * ((results[c]?.mAP50 ?? 0) / maxMAP) + 0.4 * ((results[c]?.fps ?? 0) / maxFPS)
    const scoreBest = 0.6 * ((results[best]?.mAP50 ?? 0) / maxMAP) + 0.4 * ((results[best]?.fps ?? 0) / maxFPS)
    return scoreC > scoreBest ? c : best
  })

  const cards = [
    {
      label: 'Best Accuracy',
      winner: bestAccuracy,
      icon: Target,
      color: '#22c55e',
      stat: `${((results[bestAccuracy]?.mAP50 ?? 0) * 100).toFixed(1)}% mAP@0.5`,
    },
    {
      label: 'Best Speed',
      winner: bestSpeed,
      icon: Zap,
      color: '#f59e0b',
      stat: `${(results[bestSpeed]?.fps ?? 0).toFixed(1)} FPS`,
    },
    {
      label: 'Best Trade-off',
      winner: bestTradeoff,
      icon: BarChart2,
      color: '#6366f1',
      stat: 'Balanced score (60% acc + 40% spd)',
    },
  ]

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-xl p-5">
      <h2 className="text-[#f4f4f5] font-semibold text-lg mb-4 flex items-center gap-2">
        <Trophy className="w-5 h-5 text-[#f59e0b]" />
        Winners
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {cards.map(({ label, winner, icon: Icon, color, stat }) => (
          <div
            key={label}
            className="bg-[#0a0a0a] border border-[#222222] rounded-xl p-5 text-center"
            style={{ boxShadow: `0 0 20px ${color}10` }}
          >
            <Icon className="w-8 h-8 mx-auto mb-3" style={{ color }} />
            <p className="text-[#71717a] text-xs uppercase tracking-widest mb-1">{label}</p>
            <p className="text-lg font-bold mb-1" style={{ color }}>{winner}</p>
            <p className="text-[#71717a] text-xs">{stat}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
