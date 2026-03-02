import { useEffect, useState } from 'react'
import { Loader2, ImageOff } from 'lucide-react'
import { getPreview } from '../lib/api'

const CONFIGS = ['YOLOv8', 'RT-DETR', 'NMS Ensemble', 'WBF Ensemble']

export default function FramePreview({ jobId }) {
  const [previews, setPreviews] = useState({})
  const [loading, setLoading] = useState({})

  useEffect(() => {
    if (!jobId) return

    const initial = {}
    CONFIGS.forEach(c => { initial[c] = true })
    setLoading(initial)

    CONFIGS.forEach(async (config) => {
      try {
        const data = await getPreview(jobId, config)
        setPreviews(prev => ({ ...prev, [config]: data.image }))
      } catch (e) {
        setPreviews(prev => ({ ...prev, [config]: null }))
      } finally {
        setLoading(prev => ({ ...prev, [config]: false }))
      }
    })
  }, [jobId])

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-xl p-5">
      <h2 className="text-[#f4f4f5] font-semibold text-lg mb-4">Annotated Frame Previews</h2>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {CONFIGS.map(config => (
          <div key={config} className="flex flex-col gap-2">
            <p className="text-[#71717a] text-xs text-center font-medium">{config}</p>
            <div
              className="rounded-lg overflow-hidden border border-[#222222] bg-[#0a0a0a]"
              style={{ aspectRatio: '1 / 1' }}
            >
              {loading[config] ? (
                <div className="w-full h-full flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin" style={{ color: '#6366f1' }} />
                </div>
              ) : previews[config] ? (
                <img
                  src={previews[config]}
                  alt={`${config} preview`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-2">
                  <ImageOff className="w-6 h-6" style={{ color: '#71717a' }} />
                  <span className="text-xs" style={{ color: '#71717a' }}>No preview</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
