/**
 * src/api.js — MedFill API client (updated)
 * Base URL: http://localhost:8000
 * Single endpoint: POST /api/v1/analyze
 * Returns: complete patient data + imputed lab panel in one call
 */

import axios from 'axios'

const BASE_URL = 'http://localhost:8000'

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000,   // 2-min timeout — OCR takes ~10s
  headers: { Accept: 'application/json' },
})

client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.code === 'ERR_NETWORK' || err.code === 'ECONNREFUSED') {
      const e = new Error(
        'Cannot reach MedFill backend at ' + BASE_URL +
        '. Run: uvicorn api_gateway.main:app --port 8000'
      )
      e.isOffline = true
      return Promise.reject(e)
    }
    const msg = err.response?.data?.detail ?? err.message ?? 'Unknown server error'
    return Promise.reject(new Error(msg))
  }
)

/**
 * Analyze a medical report image — one call does everything:
 *   EasyOCR → regex extraction → ML imputation
 *
 * @param {File} imageFile
 * @param {Function} [onProgress] — upload progress 0-100
 * @returns full analysis payload
 */
export async function uploadScan(imageFile, onProgress) {
  const form = new FormData()
  form.append('file', imageFile)

  const { data } = await client.post('/api/v1/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress
      ? (e) => onProgress(Math.round((e.loaded * 100) / (e.total ?? 1)))
      : undefined,
  })
  return data
}

/** Health check */
export async function checkHealth() {
  try { await client.get('/health'); return true }
  catch { return false }
}

export { BASE_URL }
