const BASE = '/api'

/**
 * Upload a video or image file for benchmarking.
 * @returns {Promise<{job_id: string}>}
 */
export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Upload failed')
  }
  return res.json()
}

/**
 * Start the benchmark run for a given job.
 * @returns {Promise<{status: string, job_id: string}>}
 */
export async function startBenchmark(jobId) {
  const res = await fetch(`${BASE}/benchmark/${jobId}`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Failed to start benchmark')
  }
  return res.json()
}

/**
 * Poll current benchmark status.
 * @returns {Promise<{progress: number, current_config: string, status: string}>}
 */
export async function getStatus(jobId) {
  const res = await fetch(`${BASE}/status/${jobId}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Failed to get status')
  }
  return res.json()
}

/**
 * Get full benchmark results for a completed job.
 * @returns {Promise<Record<string, {precision, recall, f1, mAP50, mAP5095, avg_inference_ms, fps}>>}
 */
export async function getResults(jobId) {
  const res = await fetch(`${BASE}/results/${jobId}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Failed to get results')
  }
  return res.json()
}

/**
 * Get mock sample results for UI development (no models needed).
 * @returns {Promise<Record<string, object>>}
 */
export async function getSample() {
  const res = await fetch(`${BASE}/sample`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Failed to get sample')
  }
  return res.json()
}

/**
 * Get an annotated frame preview image for a given config.
 * Config names with spaces are URL-encoded automatically.
 * @returns {Promise<{image: string}>} base64 data URL
 */
export async function getPreview(jobId, config) {
  const res = await fetch(`${BASE}/preview/${jobId}/${encodeURIComponent(config)}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || 'Failed to get preview')
  }
  return res.json()
}
