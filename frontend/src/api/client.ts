const API_BASE = '/api/v1'

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error?.message || 'Request failed')
  }

  return response.json()
}

export const api = {
  get: <T>(endpoint: string) => request<T>(endpoint),
  post: <T>(endpoint: string, data: unknown) =>
    request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  put: <T>(endpoint: string, data: unknown) =>
    request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: <T>(endpoint: string) =>
    request<T>(endpoint, {
      method: 'DELETE',
    }),
}

// Types
export interface DatasetStatus {
  name: string
  latest_date: string | null
  record_count: number
}

export interface DataStatusResponse {
  stock_id: string
  datasets: DatasetStatus[]
  checked_at: string
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

// API functions
export const systemApi = {
  health: () => api.get<HealthResponse>('/system/health'),
  dataStatus: (stockId: string = '2330') =>
    api.get<DataStatusResponse>(`/system/data-status?stock_id=${stockId}`),
}
