import { ChangeEvent, useRef } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_FILE_SIZE = 20 * 1024 * 1024

type UploadPanelProps = {
  loading: boolean
  fileName?: string
  onFileSelected: (file: File) => void
}

export function UploadPanel({ loading, fileName, onFileSelected }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      window.alert('Unsupported image format. Choose a JPG, PNG, or WEBP image.')
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      window.alert('This image is larger than the 20 MB upload limit.')
      return
    }
    onFileSelected(file)
    event.target.value = ''
  }

  return (
    <section className="upload-panel panel">
      <div>
        <span className="eyebrow">INPUT SOURCE</span>
        <h2>Inspect a board</h2>
        <p className="muted">Upload an unannotated PCB image to run the current detector.</p>
      </div>
      <div className="upload-action">
        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" onChange={handleChange} hidden />
        <button className="primary-button" onClick={() => inputRef.current?.click()} disabled={loading}>
          <span className="button-icon">＋</span>{loading ? 'Processing...' : fileName ? 'Inspect another image' : 'Choose PCB image'}
        </button>
        {fileName && <span className="file-name">{fileName}</span>}
      </div>
    </section>
  )
}
