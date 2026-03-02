import { useState, useEffect, useCallback } from 'react'
import { Car, AlertCircle } from 'lucide-react'

import UploadPanel from './components/UploadPanel'
import BenchmarkControls from './components/BenchmarkControls'
import ProgressTracker from './components/ProgressTracker'
import MetricsTable from './components/MetricsTable'
import BarChart from './components/BarChart'
import TradeOffChart from './components/TradeOffChart'
import WinnerCard from './components/WinnerCard'
import FramePreview from './components/FramePreview'

import { uploadFile, startBenchmark, getStatus, getResults, getSample } from './lib/api'

export default function App() {
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('idle') // idle | uploading | running | done | error
  const [progress, setProgress] = useState(0)
  const [currentConfig, setCurrentConfig] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  // ── Polling ────────────────────────────────────────────────────────
  useEffect(() => {
    if (status !== 'running' || !jobId) return

    const id = setInterval(async () => {
      try {
        const s = await getStatus(jobId)
        setProgress(s.progress ?? 0)
        setCurrentConfig(s.current_config ?? '')

        if (s.status === 'done') {
          clearInterval(id)
          setStatus('done')
          const r = await getResults(jobId)
          setResults(r)
        } else if (s.status === 'error') {
          clearInterval(id)
          setStatus('error')
          setError(s.error || 'Benchmark failed')
        }
      } catch (err) {
        // Network error during poll — keep trying
        console.warn('Poll error:', err.message)
      }
    }, 1500)

    return () => clearInterval(id)
  }, [status, jobId])

  // ── Handlers ───────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    setError(null)
    setStatus('uploading')
    try {
      const { job_id } = await uploadFile(file)
      setJobId(job_id)
      setStatus('idle')
      setResults(null)
      setProgress(0)
      setCurrentConfig('')
    } catch (err) {
      setStatus('error')
      setError(`Upload failed: ${err.message}`)
    }
  }, [])

  const handleRun = useCallback(async () => {
    if (!jobId) return
    setError(null)
    setResults(null)
    setProgress(0)
    setCurrentConfig('YOLOv8')
    setStatus('running')
    try {
      await startBenchmark(jobId)
    } catch (err) {
      setStatus('error')
      setError(`Failed to start benchmark: ${err.message}`)
    }
  }, [jobId])

  const handleLoadSample = useCallback(async () => {
    setError(null)
    try {
      const sample = await getSample()
      setResults(sample)
      setStatus('done')
      setProgress(100)
    } catch (err) {
      setError(`Failed to load sample: ${err.message}`)
    }
  }, [])

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen" style={{ background: '#0a0a0a' }}>
      {/* Navbar */}
      <header
        className="sticky top-0 z-10 flex items-center gap-3 px-6 py-4"
        style={{ background: '#0d0d0d', borderBottom: '1px solid #1a1a1a' }}
      >
        <Car className="w-6 h-6" style={{ color: '#6366f1' }} />
        <span className="text-[#f4f4f5] font-bold text-lg tracking-tight">CS-020</span>
        <span className="text-[#71717a] text-sm ml-1"> Vehicle Detection</span>
      </header>

      <main className="p-6 max-w-[1600px] mx-auto">
        {/* Error banner */}
        {error && (
          <div
            className="mb-6 flex items-start gap-3 px-4 py-3 rounded-lg border"
            style={{ background: '#1a0a0a', borderColor: '#7f1d1d' }}
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: '#ef4444' }} />
            <div>
              <p className="text-[#f87171] font-medium text-sm">{error}</p>
              <button
                onClick={() => setError(null)}
                className="text-[#71717a] text-xs mt-1 hover:text-[#f4f4f5] transition-colors"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-6">
          {/* ── Left sidebar ── */}
          <aside className="flex flex-col gap-5">
            <UploadPanel
              onUpload={handleUpload}
              disabled={status === 'running' || status === 'uploading'}
            />
            <BenchmarkControls
              jobId={jobId}
              status={status}
              onRun={handleRun}
              onLoadSample={handleLoadSample}
            />
            {(status === 'running' || status === 'done') && (
              <ProgressTracker
                currentConfig={currentConfig}
                progress={progress}
                status={status}
              />
            )}
          </aside>

          {/* ── Main content ── */}
          <section className="flex flex-col gap-6 min-w-0">
            {!results && status !== 'running' && (
              <div
                className="rounded-xl border border-dashed border-[#222222] p-16 text-center"
                style={{ background: '#0d0d0d' }}
              >
                <Car className="w-12 h-12 mx-auto mb-4" style={{ color: '#333333' }} />
                <p className="text-[#71717a] font-medium">No results yet</p>
                <p className="text-[#333333] text-sm mt-1">
                  Upload a file and run the benchmark, or load sample data to preview the UI
                </p>
              </div>
            )}

            {status === 'running' && !results && (
              <div
                className="rounded-xl border border-[#222222] p-16 text-center"
                style={{ background: '#0d0d0d' }}
              >
                <div
                  className="w-12 h-12 rounded-full border-2 border-t-[#6366f1] border-[#222222] mx-auto mb-4 animate-spin"
                />
                <p className="text-[#f4f4f5] font-medium">Running benchmark…</p>
                <p className="text-[#71717a] text-sm mt-1">
                  {currentConfig ? `Currently: ${currentConfig}` : 'Initialising…'} ({progress}%)
                </p>
              </div>
            )}

            {results && (
              <>
                <WinnerCard results={results} />

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <MetricsTable results={results} />
                  <div className="flex flex-col gap-6">
                    <BarChart results={results} />
                    <TradeOffChart results={results} />
                  </div>
                </div>

                {jobId && (
                  <FramePreview jobId={jobId} results={results} />
                )}
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}
