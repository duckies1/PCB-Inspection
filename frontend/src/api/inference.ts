export type BoundingBox = {
  x1: number
  y1: number
  x2: number
  y2: number
}

export type Detection = {
  class_id: number
  class_name: string
  confidence: number
  bbox: BoundingBox
}

export type DetectionResponse = {
  image: { width: number; height: number }
  classes: string[]
  detections: Detection[]
  inference: { model: string; time_ms: number; device: string }
}

type ApiError = { error?: { message?: string }; detail?: { message?: string } | string }

export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await fetch('/api/detect', { method: 'POST', body: formData })

  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as ApiError
    const detail = body.error?.message ?? body.detail
    throw new Error(typeof detail === 'string' ? detail : detail?.message ?? 'Inspection failed. Please try again.')
  }

  return response.json() as Promise<DetectionResponse>
}
