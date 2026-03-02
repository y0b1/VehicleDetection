import { useRef, useState } from 'react'
import { Upload, FileVideo, CheckCircle2 } from 'lucide-react'

const ACCEPTED = '.mp4,.avi,.mov,.mkv,.jpg,.jpeg,.png'

export default function UploadPanel({ onUpload, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [fileInfo, setFileInfo] = useState(null)

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  function handleFile(file) {
    if (!file) return
    setFileInfo({ name: file.name, size: formatBytes(file.size) })
    onUpload(file)
  }

  function onDragOver(e) {
    e.preventDefault()
    if (!disabled) setDragging(true)
  }
  function onDragLeave() { setDragging(false) }
  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }
  function onClick() {
    if (!disabled) inputRef.current?.click()
  }
  function onChange(e) {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  return (
    <div
      onClick={onClick}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200 ${
        disabled
          ? 'opacity-50 cursor-not-allowed border-[#222222] bg-[#111111]'
          : dragging
          ? 'border-[#6366f1] bg-[#6366f1]/10 cursor-copy'
          : fileInfo
          ? 'border-[#22c55e]/60 bg-[#22c55e]/5 cursor-pointer hover:border-[#22c55e]'
          : 'border-[#333333] bg-[#111111] cursor-pointer hover:border-[#6366f1]/60 hover:bg-[#6366f1]/5'
      }`}
      style={dragging ? { boxShadow: '0 0 20px #6366f130' } : {}}
    >
      {fileInfo ? (
        <>
          <CheckCircle2 className="w-10 h-10 mx-auto mb-3" style={{ color: '#22c55e' }} />
          <p className="text-[#f4f4f5] font-medium text-sm truncate px-2">{fileInfo.name}</p>
          <p className="text-[#71717a] text-xs mt-1">{fileInfo.size}</p>
          {!disabled && (
            <p className="text-[#71717a] text-xs mt-2">Click to replace</p>
          )}
        </>
      ) : (
        <>
          <Upload className="w-10 h-10 mx-auto mb-3" style={{ color: dragging ? '#6366f1' : '#71717a' }} />
          <p className="text-[#f4f4f5] font-medium text-sm">
            Drop video or images here
          </p>
          <p className="text-[#71717a] text-xs mt-1">or click to browse</p>
          <p className="text-[#333333] text-xs mt-3">mp4 · avi · mov · mkv · jpg · png</p>
        </>
      )}
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={ACCEPTED}
        onChange={onChange}
        disabled={disabled}
      />
    </div>
  )
}
