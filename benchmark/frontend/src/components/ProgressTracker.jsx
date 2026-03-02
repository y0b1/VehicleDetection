import { CheckCircle2, Loader2, Circle } from 'lucide-react'

const CONFIGS = ['YOLOv8', 'RT-DETR', 'NMS Ensemble', 'WBF Ensemble']

function getConfigStatus(config, currentConfig, overallStatus) {
  if (overallStatus === 'done') return 'done'
  if (overallStatus !== 'running') return 'pending'
  const currentIdx = CONFIGS.indexOf(currentConfig)
  const configIdx = CONFIGS.indexOf(config)
  if (configIdx < currentIdx) return 'done'
  if (configIdx === currentIdx) return 'active'
  return 'pending'
}

export default function ProgressTracker({ currentConfig, progress, status }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-[#71717a] text-xs font-medium uppercase tracking-widest mb-1">
        Progress
      </h3>
      {CONFIGS.map(config => {
        const configStatus = getConfigStatus(config, currentConfig, status)
        const currentIdx = CONFIGS.indexOf(currentConfig)
        const configIdx = CONFIGS.indexOf(config)

        let barWidth = '0%'
        if (configStatus === 'done') {
          barWidth = '100%'
        } else if (configStatus === 'active') {
          const withinConfig = ((progress - configIdx * 25) / 25) * 100
          barWidth = `${Math.max(0, Math.min(100, withinConfig))}%`
        }

        return (
          <div
            key={config}
            className="bg-[#0a0a0a] border border-[#222222] rounded-lg p-3 flex items-center gap-3"
          >
            {configStatus === 'done' && (
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" style={{ color: '#22c55e' }} />
            )}
            {configStatus === 'active' && (
              <Loader2 className="w-4 h-4 flex-shrink-0 animate-spin" style={{ color: '#6366f1' }} />
            )}
            {configStatus === 'pending' && (
              <Circle className="w-4 h-4 flex-shrink-0" style={{ color: '#333333' }} />
            )}

            <div className="flex-1 min-w-0">
              <div className="flex justify-between items-center mb-1.5">
                <span
                  className="text-sm font-medium truncate"
                  style={{ color: configStatus === 'pending' ? '#71717a' : '#f4f4f5' }}
                >
                  {config}
                </span>
                {configStatus === 'active' && (
                  <span className="text-xs ml-2 flex-shrink-0" style={{ color: '#6366f1' }}>
                    Running…
                  </span>
                )}
                {configStatus === 'done' && (
                  <span className="text-xs ml-2 flex-shrink-0" style={{ color: '#22c55e' }}>
                    Done
                  </span>
                )}
              </div>
              <div className="h-1 rounded-full" style={{ background: '#222222' }}>
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: barWidth,
                    background: configStatus === 'done' ? '#22c55e' : '#6366f1',
                    boxShadow: configStatus === 'active' ? '0 0 8px #6366f188' : 'none',
                  }}
                />
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
