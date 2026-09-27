import { useEffect, useRef, useState } from 'react'
import { Detection } from '../api/inference'

type ImageViewerProps = {
  src: string
  imageWidth: number
  imageHeight: number
  detections: Detection[]
}

type View = { scale: number; x: number; y: number }

export function ImageViewer({ src, imageWidth, imageHeight, detections }: ImageViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 })
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null)
  const [ready, setReady] = useState(false)

  function fitView() {
    const viewport = viewportRef.current
    if (!viewport) return
    const scale = Math.min((viewport.clientWidth - 48) / imageWidth, (viewport.clientHeight - 48) / imageHeight)
    setView({ scale: Math.max(scale, 0.05), x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 })
  }

  useEffect(() => {
    const image = new Image()
    image.onload = () => { imageRef.current = image; setReady(true); fitView() }
    image.src = src
    return () => { image.onload = null }
  }, [src])

  useEffect(() => {
    const observer = new ResizeObserver(() => fitView())
    if (viewportRef.current) observer.observe(viewportRef.current)
    return () => observer.disconnect()
  }, [imageWidth, imageHeight])

  useEffect(() => {
    const canvas = canvasRef.current
    const viewport = viewportRef.current
    const image = imageRef.current
    if (!canvas || !viewport || !image || !ready) return
    const ratio = window.devicePixelRatio || 1
    canvas.width = viewport.clientWidth * ratio
    canvas.height = viewport.clientHeight * ratio
    canvas.style.width = `${viewport.clientWidth}px`
    canvas.style.height = `${viewport.clientHeight}px`
    const context = canvas.getContext('2d')
    if (!context) return
    context.setTransform(ratio, 0, 0, ratio, 0, 0)
    context.clearRect(0, 0, viewport.clientWidth, viewport.clientHeight)
    context.save()
    context.translate(view.x, view.y)
    context.scale(view.scale, view.scale)
    context.translate(-imageWidth / 2, -imageHeight / 2)
    context.drawImage(image, 0, 0, imageWidth, imageHeight)

    detections.forEach((detection) => {
      const { x1, y1, x2, y2 } = detection.bbox
      const width = x2 - x1
      const height = y2 - y1
      context.strokeStyle = '#e6ff68'
      context.lineWidth = 2 / view.scale
      context.strokeRect(x1, y1, width, height)
      const label = `${detection.class_name.replaceAll('_', ' ')}  ${(detection.confidence * 100).toFixed(0)}%`
      context.font = `${Math.max(12 / view.scale, 9)}px ui-monospace, monospace`
      const labelWidth = context.measureText(label).width + 12 / view.scale
      context.fillStyle = '#e6ff68'
      context.fillRect(x1, Math.max(0, y1 - 22 / view.scale), labelWidth, 22 / view.scale)
      context.fillStyle = '#10221f'
      context.fillText(label, x1 + 6 / view.scale, Math.max(15 / view.scale, y1 - 7 / view.scale))
    })
    context.restore()
  }, [detections, imageWidth, imageHeight, view, ready])

  function handleWheel(event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault()
    const direction = event.deltaY > 0 ? 0.9 : 1.1
    setView((current) => ({ ...current, scale: Math.min(12, Math.max(0.03, current.scale * direction)) }))
  }

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId)
    setDrag({ x: event.clientX - view.x, y: event.clientY - view.y })
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag) return
    setView((current) => ({ ...current, x: event.clientX - drag.x, y: event.clientY - drag.y }))
  }

  return (
    <section className="viewer panel">
      <div className="viewer-heading"><div><span className="eyebrow">VISUAL INSPECTION</span><h2>Board view</h2></div><button className="quiet-button" onClick={fitView}>Reset view</button></div>
      <div className="viewer-viewport" ref={viewportRef} onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
        <canvas ref={canvasRef} />
        {!ready && <div className="viewer-placeholder">Loading image...</div>}
        {ready && detections.length === 0 && <div className="viewer-status">No visible detections</div>}
      </div>
      <div className="viewer-footer"><span>Scroll to zoom · drag to pan</span><span>{Math.round(view.scale * 100)}%</span></div>
    </section>
  )
}
