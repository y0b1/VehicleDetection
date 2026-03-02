const CONFIGS = ['YOLOv8', 'YOLOv9', 'NMS Ensemble', 'WBF Ensemble']

const METRICS = [
  { key: 'precision',        label: 'Precision',        higherBetter: true,  format: v => `${(v * 100).toFixed(1)}%` },
  { key: 'recall',           label: 'Recall',           higherBetter: true,  format: v => `${(v * 100).toFixed(1)}%` },
  { key: 'f1',               label: 'F1 Score',         higherBetter: true,  format: v => `${(v * 100).toFixed(1)}%` },
  { key: 'mAP50',            label: 'mAP@0.5',          higherBetter: true,  format: v => `${(v * 100).toFixed(1)}%` },
  { key: 'mAP5095',          label: 'mAP@[.5:.95]',     higherBetter: true,  format: v => `${(v * 100).toFixed(1)}%` },
  { key: 'avg_inference_ms', label: 'Inference (ms)',   higherBetter: false, format: v => `${v.toFixed(1)} ms` },
  { key: 'fps',              label: 'FPS',              higherBetter: true,  format: v => v.toFixed(1) },
]

export default function MetricsTable({ results }) {
  function getBestConfig(metricKey, higherBetter) {
    let best = null
    let bestVal = higherBetter ? -Infinity : Infinity
    for (const config of CONFIGS) {
      const val = results[config]?.[metricKey]
      if (val == null) continue
      if (higherBetter ? val > bestVal : val < bestVal) {
        bestVal = val
        best = config
      }
    }
    return best
  }

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-xl p-5 overflow-x-auto">
      <h2 className="text-[#f4f4f5] font-semibold text-lg mb-4">Metrics Comparison</h2>
      <table className="w-full text-sm min-w-[600px]">
        <thead>
          <tr className="border-b border-[#222222]">
            <th className="text-left py-2 px-3 text-[#71717a] font-medium">Metric</th>
            {CONFIGS.map(c => (
              <th key={c} className="text-right py-2 px-3 text-[#71717a] font-medium whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRICS.map(({ key, label, higherBetter, format }, rowIdx) => {
            const bestConfig = getBestConfig(key, higherBetter)
            return (
              <tr
                key={key}
                className={rowIdx % 2 === 0 ? 'bg-[#111111]' : 'bg-[#0f0f0f]'}
              >
                <td className="py-2.5 px-3 text-[#71717a]">{label}</td>
                {CONFIGS.map(config => {
                  const val = results[config]?.[key]
                  const isBest = config === bestConfig && val != null
                  return (
                    <td
                      key={config}
                      className={`py-2.5 px-3 text-right font-mono ${
                        isBest
                          ? 'text-[#22c55e] font-bold bg-[#22c55e]/5'
                          : 'text-[#f4f4f5]'
                      }`}
                    >
                      {val != null ? format(val) : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
