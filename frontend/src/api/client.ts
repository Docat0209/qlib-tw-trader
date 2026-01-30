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
  patch: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    }),
  delete: (endpoint: string) =>
    fetch(`${API_BASE}${endpoint}`, { method: 'DELETE' }),
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

// Factor Types
export interface Factor {
  id: string
  name: string
  display_name: string | null
  category: string
  description: string | null
  formula: string
  selection_rate: number
  times_selected: number
  times_evaluated: number
  enabled: boolean
  created_at: string
}

export interface FactorDetail extends Factor {
  selection_history: {
    model_id: string
    trained_at: string
    selected: boolean
  }[]
}

export interface FactorListResponse {
  items: Factor[]
  total: number
}

export interface FactorCreate {
  name: string
  display_name?: string
  category?: string
  description?: string
  formula: string
}

export interface FactorUpdate {
  name?: string
  display_name?: string
  category?: string
  description?: string
  formula?: string
}

// API functions
export const systemApi = {
  health: () => api.get<HealthResponse>('/system/health'),
  dataStatus: (stockId: string = '2330') =>
    api.get<DataStatusResponse>(`/system/data-status?stock_id=${stockId}`),
}

export const factorApi = {
  list: (category?: string, enabled?: boolean) => {
    const params = new URLSearchParams()
    if (category) params.set('category', category)
    if (enabled !== undefined) params.set('enabled', String(enabled))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<FactorListResponse>(`/factors${query}`)
  },
  get: (id: number) => api.get<FactorDetail>(`/factors/${id}`),
  create: (data: FactorCreate) => api.post<Factor>('/factors', data),
  update: (id: number, data: FactorUpdate) => api.put<Factor>(`/factors/${id}`, data),
  delete: (id: number) => api.delete(`/factors/${id}`),
  toggle: (id: number) => api.patch<Factor>(`/factors/${id}/toggle`),
}

// Model Types
export interface Period {
  start: string
  end: string
}

export interface ModelMetrics {
  ic: number | null
  icir: number | null
}

export interface Model {
  id: string
  trained_at: string
  factor_count: number | null
  factors: string[]
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  training_duration_seconds: number | null
  is_current: boolean
}

export interface ModelHistoryItem {
  id: string
  trained_at: string
  factor_count: number | null
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  is_current: boolean
}

export interface ModelHistoryResponse {
  items: ModelHistoryItem[]
  total: number
}

export interface ModelComparisonItem {
  id: string
  trained_at: string
  ic: number | null
  icir: number | null
  factor_count: number | null
}

export interface ModelComparisonResponse {
  models: ModelComparisonItem[]
}

export interface ModelStatus {
  last_trained_at: string | null
  days_since_training: number | null
  needs_retrain: boolean
  retrain_threshold_days: number
  current_job: number | null
}

export interface TrainRequest {
  train_end: string
  valid_end: string
}

export interface TrainResponse {
  job_id: string
  status: string
  message: string
}

export const modelApi = {
  current: () => api.get<Model>('/models/current'),
  history: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<ModelHistoryResponse>(`/models/history${query}`)
  },
  comparison: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<ModelComparisonResponse>(`/models/comparison${query}`)
  },
  status: () => api.get<ModelStatus>('/models/status'),
  train: (data: TrainRequest) => api.post<TrainResponse>('/models/train', data),
}

// Portfolio Types
export interface PositionItem {
  symbol: string
  name: string | null
  shares: number
  avg_cost: number
  current_price: number | null
  market_value: number | null
  unrealized_pnl: number | null
  unrealized_pnl_pct: number | null
  weight: number | null
}

export interface PositionsResponse {
  as_of: string
  total_value: number
  cash: number
  positions: PositionItem[]
}

export interface PredictionSignal {
  symbol: string
  name: string | null
  score: number
  rank: number
  signal: string
  current_position: number | null
}

export interface PredictionsLatestResponse {
  date: string
  model_id: string
  signals: PredictionSignal[]
}

export interface PredictionHistoryItem {
  date: string
  symbol: string
  score: number
  rank: number
  signal: string
}

export interface PredictionsHistoryResponse {
  items: PredictionHistoryItem[]
  total: number
}

export interface TradeItem {
  id: number
  date: string
  symbol: string
  name: string | null
  side: string
  shares: number
  price: number
  amount: number
  commission: number
  reason: string | null
}

export interface TradesResponse {
  items: TradeItem[]
  total: number
}

export const portfolioApi = {
  positions: () => api.get<PositionsResponse>('/positions'),
  predictionsLatest: () => api.get<PredictionsLatestResponse>('/predictions/latest'),
  predictionsHistory: (params?: { start_date?: string; end_date?: string; symbol?: string }) => {
    const searchParams = new URLSearchParams()
    if (params?.start_date) searchParams.set('start_date', params.start_date)
    if (params?.end_date) searchParams.set('end_date', params.end_date)
    if (params?.symbol) searchParams.set('symbol', params.symbol)
    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return api.get<PredictionsHistoryResponse>(`/predictions/history${query}`)
  },
  trades: (params?: { start_date?: string; end_date?: string; symbol?: string }) => {
    const searchParams = new URLSearchParams()
    if (params?.start_date) searchParams.set('start_date', params.start_date)
    if (params?.end_date) searchParams.set('end_date', params.end_date)
    if (params?.symbol) searchParams.set('symbol', params.symbol)
    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return api.get<TradesResponse>(`/trades${query}`)
  },
}

// Performance Types
export interface Returns {
  today: number | null
  wtd: number | null
  mtd: number | null
  ytd: number | null
  total: number | null
}

export interface PerformanceSummary {
  as_of: string
  returns: Returns
  benchmark_returns: Returns
  alpha: {
    mtd: number | null
    ytd: number | null
  }
}

export interface EquityCurvePoint {
  date: string
  portfolio_value: number
  cumulative_return: number
  benchmark_return: number | null
}

export interface MonthlyReturn {
  year: number
  month: number
  return: number
  benchmark: number | null
}

export const performanceApi = {
  summary: () => api.get<PerformanceSummary>('/performance/summary'),
  equityCurve: (params?: { start_date?: string; end_date?: string; benchmark?: boolean }) => {
    const searchParams = new URLSearchParams()
    if (params?.start_date) searchParams.set('start_date', params.start_date)
    if (params?.end_date) searchParams.set('end_date', params.end_date)
    if (params?.benchmark !== undefined) searchParams.set('benchmark', String(params.benchmark))
    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return api.get<{ data: EquityCurvePoint[] }>(`/performance/equity-curve${query}`)
  },
  monthly: () => api.get<{ data: MonthlyReturn[] }>('/performance/monthly'),
}

// Dashboard Types
export interface DashboardSummary {
  factors: {
    total: number
    enabled: number
    low_selection_count: number
  }
  model: {
    last_trained_at: string | null
    days_since_training: number | null
    needs_retrain: boolean
    factor_count: number | null
    ic: number | null
    icir: number | null
  }
  prediction: {
    date: string | null
    buy_signals: number
    sell_signals: number
    top_pick: { symbol: string; score: number } | null
  }
  data_status: {
    is_complete: boolean
    last_updated: string | null
    missing_count: number
  }
  performance: {
    today_return: number | null
    mtd_return: number | null
    ytd_return: number | null
    total_return: number | null
  }
}

export const dashboardApi = {
  summary: () => api.get<DashboardSummary>('/dashboard/summary'),
}
