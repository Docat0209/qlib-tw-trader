const API_BASE = '/api/v1'

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    // FastAPI 使用 detail，其他可能用 error.message
    const message = error.detail || error.error?.message || error.message || `HTTP ${response.status}`
    throw new Error(message)
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
  earliest_date: string | null
  latest_date: string | null
  is_fresh: boolean
}

export interface StockItem {
  stock_id: string
  is_fresh: boolean
}

export interface DataStatusResponse {
  datasets: DatasetStatus[]
  stocks: StockItem[]
  checked_at: string
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

export interface SyncRequest {
  stock_id: string
  start_date: string
  end_date: string
  datasets?: string[]
}

export interface SyncResult {
  dataset: string
  records_fetched: number
  success: boolean
  error: string | null
}

export interface SyncResponse {
  stock_id: string
  results: SyncResult[]
  total_records: number
  synced_at: string
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
  dataStatus: () => api.get<DataStatusResponse>('/system/data-status'),
  sync: (data: SyncRequest) => api.post<SyncResponse>('/system/sync', data),
}

export interface ValidateResponse {
  valid: boolean
  error?: string
  fields_used: string[]
  operators_used: string[]
  warnings: string[]
}

export interface SeedResponse {
  success: boolean
  inserted: number
  message: string
}

export interface AvailableFieldsResponse {
  fields: string[]
  operators: string[]
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
  validate: (expression: string) => api.post<ValidateResponse>('/factors/validate', { expression }),
  seed: (force?: boolean) => {
    const query = force ? '?force=true' : ''
    return api.post<SeedResponse>(`/factors/seed${query}`, {})
  },
  available: () => api.get<AvailableFieldsResponse>('/factors/available'),
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

export interface FactorSummary {
  id: string
  name: string
  display_name: string | null
  category: string
  ic_value: number | null
}

export interface SelectionInfo {
  method: string | null
  config: Record<string, unknown> | null
  stats: Record<string, unknown> | null
}

export interface Model {
  id: string
  name: string | null
  description: string | null
  status: string
  trained_at: string
  factor_count: number | null
  factors: string[]
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  training_duration_seconds: number | null
  candidate_factors: FactorSummary[]
  selected_factors: FactorSummary[]
  selection: SelectionInfo | null
}

export interface ModelSummary {
  id: string
  name: string | null
  status: string
  trained_at: string
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  factor_count: number | null
  candidate_count: number | null
  selection_method: string | null
}

export interface ModelListResponse {
  items: ModelSummary[]
  total: number
}

// 向後兼容
export type ModelHistoryItem = ModelSummary

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
  train_end?: string
  hyperparams_id?: number
}

export interface TrainResponse {
  job_id: string
  status: string
  message: string
}

export interface DataRangeResponse {
  start: string
  end: string
}

export interface DeleteResponse {
  status: string
  id: string
}

export interface SetCurrentResponse {
  status: string
  id: string
}

export const modelApi = {
  // 新增的方法
  list: () => api.get<ModelListResponse>('/models'),
  get: (id: string) => api.get<Model>(`/models/${id}`),
  delete: (id: string) => api.delete(`/models/${id}`).then(res => res.json() as Promise<DeleteResponse>),
  setCurrent: (id: string) => api.patch<SetCurrentResponse>(`/models/${id}/current`),
  dataRange: () => api.get<DataRangeResponse>('/models/data-range'),

  // 現有方法（向後兼容）
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
  train: (data?: TrainRequest) => api.post<TrainResponse>('/models/train', data || {}),
}

// Hyperparams Types
export interface CultivationPeriod {
  train_start: string
  train_end: string
  valid_start: string
  valid_end: string
  best_ic: number
  params: Record<string, number>
}

export interface HyperparamsSummary {
  id: number
  name: string
  cultivated_at: string
  n_periods: number
  learning_rate: number | null
  num_leaves: number | null
}

export interface HyperparamsDetail extends HyperparamsSummary {
  params: Record<string, number>
  stability: Record<string, number>
  periods: CultivationPeriod[]
}

export interface HyperparamsListResponse {
  items: HyperparamsSummary[]
  total: number
}

export interface CultivateRequest {
  name: string
  n_periods?: number
  n_trials_per_period?: number
}

export interface CultivateResponse {
  job_id: string
  status: string
  message: string
}

export const hyperparamsApi = {
  list: () => api.get<HyperparamsListResponse>('/hyperparams'),
  get: (id: number) => api.get<HyperparamsDetail>(`/hyperparams/${id}`),
  cultivate: (data: CultivateRequest) => api.post<CultivateResponse>('/hyperparams', data),
  update: (id: number, data: { name: string }) => api.patch<HyperparamsDetail>(`/hyperparams/${id}`, data),
  delete: (id: number) => api.delete(`/hyperparams/${id}`).then(res => res.json() as Promise<{ status: string; id: number }>),
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

export interface PredictionRequest {
  model_id: number
  top_k: number
  trade_date?: string  // 預計交易日期，YYYY-MM-DD 格式，null = 最新資料日期的下一天
}

export interface PredictionSignal {
  rank: number
  symbol: string
  name: string | null
  score: number
}

export interface PredictionsResponse {
  trade_date: string  // 預計交易日期
  feature_date: string  // 實際使用的特徵資料日期
  model_name: string
  signals: PredictionSignal[]
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
  generatePredictions: (data: PredictionRequest) =>
    api.post<PredictionsResponse>('/predictions/generate', data),
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

// Job Types
export interface JobItem {
  id: string
  job_type: string
  status: string
  progress: number
  message: string | null
  started_at: string | null
  completed_at: string | null
}

export interface JobDetail extends JobItem {
  result: string | null
}

export interface JobListResponse {
  items: JobItem[]
  total: number
}

export const jobApi = {
  list: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<JobListResponse>(`/jobs${query}`)
  },
  get: (jobId: string) => api.get<JobDetail>(`/jobs/${jobId}`),
  cancel: (jobId: string) => api.delete(`/jobs/${jobId}`).then(res => res.json() as Promise<{ status: string; id: string }>),
}

// Backtest Types
export interface BacktestMetrics {
  // 新格式
  total_return_with_cost: number | null
  total_return_without_cost: number | null
  annual_return_with_cost: number | null
  annual_return_without_cost: number | null
  sharpe_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  total_trades: number | null
  total_cost: number | null
  // 向後兼容
  total_return: number | null
  annual_return: number | null
  profit_factor: number | null
}

export interface EquityCurvePoint {
  date: string
  equity: number
  benchmark: number | null
  drawdown: number | null
}

export interface BacktestItem {
  id: number
  model_id: number
  start_date: string
  end_date: string
  initial_capital: number
  max_positions: number
  status: string
  metrics: BacktestMetrics | null
  created_at: string
}

export interface BacktestDetail extends BacktestItem {
  equity_curve: EquityCurvePoint[] | null
}

export interface BacktestListResponse {
  items: BacktestItem[]
  total: number
}

export interface BacktestRequest {
  model_id: number
  initial_capital?: number
  max_positions?: number
  trade_price?: 'close' | 'open'  // 交易價格：收盤價或開盤價
}

export interface BacktestRunResponse {
  backtest_id: number
  job_id: string
  status: string
  message: string
}

export interface StockTradeInfo {
  stock_id: string
  name: string
  buy_count: number
  sell_count: number
  total_pnl: number | null
}

export interface StockTradeListResponse {
  backtest_id: number
  items: StockTradeInfo[]
  total: number
}

export interface KlinePoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface TradePoint {
  date: string
  side: 'buy' | 'sell'
  price: number
  shares: number
  amount?: number
  commission?: number
  pnl?: number | null  // 賣出時的盈虧金額
  pnl_pct?: number | null  // 賣出時的盈虧 %
  holding_days?: number | null  // 持有天數
  stock_id?: string  // 股票代碼（全局交易列表用）
  stock_name?: string  // 股票名稱
}

export interface AllTradesResponse {
  backtest_id: number
  items: TradePoint[]
  total_pnl: number  // 已實現盈虧
  unrealized_pnl: number  // 未實現盈虧（持倉）
  total_equity_pnl: number  // 總計（已實現 + 未實現）
  total: number
}

export interface StockKlineResponse {
  stock_id: string
  name: string
  klines: KlinePoint[]
  trades: TradePoint[]
}

export const backtestApi = {
  list: (modelId?: number, limit?: number) => {
    const params = new URLSearchParams()
    if (modelId) params.set('model_id', String(modelId))
    if (limit) params.set('limit', String(limit))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<BacktestListResponse>(`/backtest${query}`)
  },
  get: (backtestId: number) => api.get<BacktestDetail>(`/backtest/${backtestId}`),
  run: (data: BacktestRequest) => api.post<BacktestRunResponse>('/backtest/run', data),
  getStocks: (backtestId: number) => api.get<StockTradeListResponse>(`/backtest/${backtestId}/stocks`),
  getStockKline: (backtestId: number, stockId: string) =>
    api.get<StockKlineResponse>(`/backtest/${backtestId}/stocks/${stockId}`),
  getAllTrades: (backtestId: number) => api.get<AllTradesResponse>(`/backtest/${backtestId}/trades`),
  delete: (backtestId: number) => api.delete(`/backtest/${backtestId}`),
}

// Dataset Types
export interface DatasetInfo {
  name: string
  display_name: string
  category: string
  source: string
  status: 'available' | 'needs_accumulation' | 'not_implemented' | 'pending'
  description: string | null
  requires_stock_id: boolean
}

export interface DatasetListResponse {
  datasets: DatasetInfo[]
  total: number
}

export interface TestResult {
  dataset: string
  success: boolean
  record_count: number
  sample_data: Record<string, unknown>[] | null
  error: string | null
}

export interface CategoryInfo {
  id: string
  name: string
  total: number
  available: number
}

// Universe Types
export interface StockInfo {
  stock_id: string
  name: string
  market_cap: number
  rank: number
}

export interface UniverseResponse {
  name: string
  description: string
  total: number
  stocks: StockInfo[]
  updated_at: string | null
}

export interface UniverseStats {
  total: number
  min_market_cap: number
  max_market_cap: number
  updated_at: string | null
}

export const universeApi = {
  get: () => api.get<UniverseResponse>('/universe'),
  stats: () => api.get<UniverseStats>('/universe/stats'),
  ids: () => api.get<{ stock_ids: string[]; total: number }>('/universe/ids'),
  sync: () => api.post<{ success: boolean; total: number; updated_at: string }>('/universe/sync', {}),
}

export const datasetsApi = {
  list: (category?: string, status?: string) => {
    const params = new URLSearchParams()
    if (category) params.set('category', category)
    if (status) params.set('status', status)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<DatasetListResponse>(`/datasets${query}`)
  },
  test: (datasetName: string, stockId?: string, days?: number) => {
    const params = new URLSearchParams()
    if (stockId) params.set('stock_id', stockId)
    if (days) params.set('days', String(days))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<TestResult>(`/datasets/test/${datasetName}${query}`)
  },
  categories: () => api.get<{ categories: CategoryInfo[] }>('/datasets/categories'),
}

// Sync Types
export interface SyncStatusItem {
  stock_id: string
  name: string
  rank: number
  earliest_date: string | null
  latest_date: string | null
  total_records: number
  missing_count: number
  coverage_pct: number
}

export interface SyncStatusResponse {
  trading_days: number
  start_date: string
  end_date: string
  stocks: SyncStatusItem[]
}

export interface SyncCalendarResponse {
  start_date: string
  end_date: string
  new_dates: number
  total_dates: number
}

export interface SyncStockResponse {
  stock_id: string
  fetched: number
  inserted: number
  missing_dates: string[]
}

export interface SyncBulkResponse {
  date: string
  total: number
  inserted: number
  days_synced?: number
  error: string | null
}

export interface SyncAllResponse {
  stocks: number
  total_inserted: number
  errors: { stock_id: string; error: string }[]
}

// 月營收專用
export interface MonthlyStatusItem {
  stock_id: string
  name: string
  rank: number
  earliest_month: string | null
  latest_month: string | null
  total_records: number
  missing_count: number
  coverage_pct: number
}

export interface MonthlyStatusResponse {
  expected_months: number
  start_year: number
  end_year: number
  stocks: MonthlyStatusItem[]
}

export interface MonthlyStockResponse {
  stock_id: string
  fetched: number
  inserted: number
  missing_months: string[]
}

export const syncApi = {
  status: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/status${query}`)
  },
  calendar: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncCalendarResponse>(`/sync/calendar${query}`, {})
  },
  stock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/stock/${stockId}${query}`, {})
  },
  bulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/bulk${query}`, {})
  },
  all: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/all${query}`, {})
  },
  // PER/PBR/殖利率
  perStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/per/status${query}`)
  },
  perBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/per/bulk${query}`, {})
  },
  perStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/per/stock/${stockId}${query}`, {})
  },
  perAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/per/all${query}`, {})
  },
  // 三大法人
  institutionalStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/institutional/status${query}`)
  },
  institutionalBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/institutional/bulk${query}`, {})
  },
  institutionalStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/institutional/stock/${stockId}${query}`, {})
  },
  institutionalAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/institutional/all${query}`, {})
  },
  // 融資融券
  marginStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/margin/status${query}`)
  },
  marginBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/margin/bulk${query}`, {})
  },
  marginStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/margin/stock/${stockId}${query}`, {})
  },
  marginAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/margin/all${query}`, {})
  },
  // 還原股價 (yfinance)
  adjStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/adj/status${query}`)
  },
  adjBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/adj/bulk${query}`, {})
  },
  adjStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/adj/stock/${stockId}${query}`, {})
  },
  adjAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/adj/all${query}`, {})
  },
  // 外資持股
  shareholdingStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/shareholding/status${query}`)
  },
  shareholdingBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/shareholding/bulk${query}`, {})
  },
  shareholdingStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/shareholding/stock/${stockId}${query}`, {})
  },
  shareholdingAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/shareholding/all${query}`, {})
  },
  // 借券明細
  securitiesLendingStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/securities-lending/status${query}`)
  },
  securitiesLendingStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/securities-lending/stock/${stockId}${query}`, {})
  },
  securitiesLendingAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/securities-lending/all${query}`, {})
  },
  // 月營收
  monthlyRevenueStatus: (startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<MonthlyStatusResponse>(`/sync/monthly-revenue/status${query}`)
  },
  monthlyRevenueStock: (stockId: string, startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<MonthlyStockResponse>(`/sync/monthly-revenue/stock/${stockId}${query}`, {})
  },
  monthlyRevenueAll: (startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/monthly-revenue/all${query}`, {})
  },
}
