import { Detection } from '../api/inference'

function labelForClass(name: string) {
  return name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

type DefectSummaryProps = {
  detections: Detection[]
  classes: string[]
  enabledClasses: Set<string>
  onToggleClass: (className: string) => void
  threshold: number
  onThresholdChange: (threshold: number) => void
}

export function DefectSummary({ detections, classes, enabledClasses, onToggleClass, threshold, onThresholdChange }: DefectSummaryProps) {
  const visible = detections.filter((detection) => detection.confidence >= threshold && enabledClasses.has(detection.class_name))
  const total = visible.length

  return (
    <section className="panel summary-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">SIGNAL BREAKDOWN</span>
          <h2>Defect summary</h2>
        </div>
        <div className="total-count"><strong>{total}</strong><span>visible defects</span></div>
      </div>
      <div className="threshold-control">
        <div className="control-label"><span>Confidence threshold</span><output>{threshold.toFixed(2)}</output></div>
        <input type="range" min="0" max="0.95" step="0.05" value={threshold} onChange={(event) => onThresholdChange(Number(event.target.value))} />
        <div className="range-labels"><span>All predictions</span><span>Strict</span></div>
      </div>
      <div className="class-list">
        {classes.map((className) => {
          const count = visible.filter((detection) => detection.class_name === className).length
          const enabled = enabledClasses.has(className)
          return (
            <label className={`class-row ${enabled ? '' : 'is-disabled'}`} key={className}>
              <input type="checkbox" checked={enabled} onChange={() => onToggleClass(className)} />
              <span className="checkmark" />
              <span className="class-name">{labelForClass(className)}</span>
              <span className="class-count">{count.toString().padStart(2, '0')}</span>
            </label>
          )
        })}
      </div>
      {detections.length > 0 && total === 0 && <p className="empty-note">No defects detected above the current filters.</p>}
    </section>
  )
}
