import { Play, FlaskConical, Loader2 } from 'lucide-react'

export default function BenchmarkControls({ jobId, status, onRun, onLoadSample }) {
  const isRunning = status === 'running'
  const canRun = !!jobId && !isRunning

  return (
    <div className="flex flex-col gap-3">
      <button
        onClick={onRun}
        disabled={!canRun}
        className="w-full py-3 px-6 rounded-lg font-semibold text-white text-sm
                   flex items-center justify-center gap-2 transition-all duration-200"
        style={{
          background: canRun ? '#6366f1' : '#333333',
          cursor: canRun ? 'pointer' : 'not-allowed',
          opacity: canRun ? 1 : 0.5,
          boxShadow: canRun ? '0 0 20px #6366f140' : 'none',
        }}
      >
        {isRunning ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Play className="w-4 h-4" />
        )}
        {isRunning ? 'Benchmark Running…' : 'Run Benchmark'}
      </button>

      <button
        onClick={onLoadSample}
        disabled={isRunning}
        className="w-full py-2.5 px-6 rounded-lg text-sm font-medium
                   flex items-center justify-center gap-2 transition-all duration-200
                   border border-[#333333] hover:border-[#6366f1]/50 hover:bg-[#6366f1]/5"
        style={{
          color: isRunning ? '#71717a' : '#a1a1aa',
          cursor: isRunning ? 'not-allowed' : 'pointer',
        }}
      >
        <FlaskConical className="w-4 h-4" />
        Load Sample Data
      </button>

      {!jobId && (
        <p className="text-[#71717a] text-xs text-center">
          Upload a file first to enable benchmark
        </p>
      )}
      {jobId && !isRunning && (
        <p className="text-[#71717a] text-xs text-center">
          Runs 4 configurations sequentially
        </p>
      )}
    </div>
  )
}
