import { useEffect, useMemo, useState } from 'react'
import { DetectionResponse, detectImage } from '../api/inference'
import { DefectSummary } from '../components/DefectSummary'
import { ImageViewer } from '../components/ImageViewer'
import { UploadPanel } from '../components/UploadPanel'

export function Dashboard() {
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [imageUrl, setImageUrl] = useState('')
  const [fileName, setFileName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [threshold, setThreshold] = useState(0.5)
  const [enabledClasses, setEnabledClasses] = useState<Set<string>>(new Set())

  const classes = useMemo(() => result?.classes ?? [], [result])
  const visibleDetections = useMemo(() => result?.detections.filter((detection) => detection.confidence >= threshold && enabledClasses.has(detection.class_name)) ?? [], [result, threshold, enabledClasses])

  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl) }, [imageUrl])

  async function handleFileSelected(file: File) {
    setLoading(true)
    setError('')
    try {
      const nextResult = await detectImage(file)
      if (imageUrl) URL.revokeObjectURL(imageUrl)
      setImageUrl(URL.createObjectURL(file))
      setFileName(file.name)
      setResult(nextResult)
      setThreshold(0.5)
      setEnabledClasses(new Set(nextResult.detections.map((detection) => detection.class_name)))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to connect to the inspection server.')
    } finally {
      setLoading(false)
    }
  }

  function toggleClass(className: string) {
    setEnabledClasses((current) => {
      const next = new Set(current)
      if (next.has(className)) next.delete(className)
      else next.add(className)
      return next
    })
  }

  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand"><span className="brand-mark">PI</span><span>PCB inspection</span></div><div className="status-chip"><span className="status-dot" /> inference ready</div></header>
      <div className="content-grid">
        <section className="intro"><div><span className="eyebrow">ENGINEERING CONSOLE / 01</span><h1>Find the fault<br /><em>before it ships.</em></h1><p>Interactive visual inspection for high-density circuit boards. Upload a board, then interrogate every predicted defect.</p></div>{result && <div className="run-stamp"><span>LAST RUN</span><strong>{result.inference.time_ms.toFixed(0)} ms</strong><small>{result.inference.model} · {result.inference.device}</small></div>}</section>
        <UploadPanel loading={loading} fileName={fileName} onFileSelected={handleFileSelected} />
        {error && <div className="error-banner"><strong>Inspection unavailable</strong><span>{error}</span></div>}
        {result && imageUrl ? <>
          <ImageViewer src={imageUrl} imageWidth={result.image.width} imageHeight={result.image.height} detections={visibleDetections} />
          <aside className="side-column"><DefectSummary detections={result.detections} classes={classes} enabledClasses={enabledClasses} onToggleClass={toggleClass} threshold={threshold} onThresholdChange={setThreshold} /><section className="panel metadata"><span className="eyebrow">RUN METADATA</span><h2>Inference information</h2><dl><div><dt>Model</dt><dd>{result.inference.model}</dd></div><div><dt>Image resolution</dt><dd>{result.image.width} × {result.image.height}</dd></div><div><dt>Device</dt><dd>{result.inference.device}</dd></div><div><dt>Raw detections</dt><dd>{result.detections.length.toString().padStart(2, '0')}</dd></div></dl></section></aside>
        </> : <section className="empty-state panel"><div className="crosshair">＋</div><h2>Waiting for an image</h2><p>Your original board image and live detection overlays will appear here.</p></section>}
      </div>
      <footer><span>PCB INSPECTION SYSTEM</span><span>MODEL-INDEPENDENT DETECTION CONTRACT</span></footer>
    </main>
  )
}
