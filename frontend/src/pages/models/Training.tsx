import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Play,
  Clock,
  BarChart3,
  Layers,
  CheckCircle,
  Activity,
  Loader2,
  RefreshCw,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Trash2,
  Star,
  XCircle,
} from 'lucide-react'
import { modelApi, ModelSummary, ModelStatus, Model, FactorSummary, DataRangeResponse, hyperparamsApi, HyperparamsSummary } from '@/api/client'
import { Link } from 'react-router-dom'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'

export function Training() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [expandedModel, setExpandedModel] = useState<Model | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [dataRange, setDataRange] = useState<DataRangeResponse | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<string>('')
  const [hyperparamsList, setHyperparamsList] = useState<HyperparamsSummary[]>([])
  const [selectedHpId, setSelectedHpId] = useState<number | null>(null)

  // WebSocket 訓練進度追蹤
  const { activeJob, clearJob, cancelJob, isConnected } = useJobs()

  // 訓練狀態
  const isTraining = activeJob?.job_type === 'train' &&
    ['queued', 'running'].includes(activeJob.status)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [listRes, statusRes, rangeRes, hpRes] = await Promise.all([
        modelApi.list(),
        modelApi.status(),
        modelApi.dataRange().catch(() => null),
        hyperparamsApi.list().catch(() => ({ items: [] })),
      ])
      setModels(listRes.items)
      setStatus(statusRes)
      if (rangeRes) {
        setDataRange(rangeRes)
        // 預設選擇最新的驗證結束月份（資料結束日期的月份）
        if (!selectedMonth) {
          const endDate = new Date(rangeRes.end)
          const defaultMonth = `${endDate.getFullYear()}-${String(endDate.getMonth() + 1).padStart(2, '0')}`
          setSelectedMonth(defaultMonth)
        }
      }
      // 載入超參數列表並預設選擇 current
      setHyperparamsList(hpRes.items)
      if (selectedHpId === null && hpRes.items.length > 0) {
        const current = hpRes.items.find(hp => hp.is_current)
        setSelectedHpId(current?.id ?? hpRes.items[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [selectedMonth, selectedHpId])

  // 初始載入
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // 自動刷新（監聯 data_updated 事件）
  useFetchOnChange('models', fetchData)
  useFetchOnChange('hyperparams', fetchData)

  // 監聽訓練完成，自動刷新列表
  useEffect(() => {
    if (activeJob?.status === 'completed' && activeJob?.job_type === 'train') {
      fetchData()
      // 延遲清除 job，讓用戶看到完成狀態
      const timer = setTimeout(() => clearJob(activeJob.id), 5000)
      return () => clearTimeout(timer)
    }
    if (activeJob?.status === 'failed' && activeJob?.job_type === 'train') {
      // 失敗後也刷新列表
      fetchData()
    }
  }, [activeJob?.status, activeJob?.job_type, activeJob?.id, clearJob, fetchData])

  const handleExpand = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null)
      setExpandedModel(null)
      return
    }

    setExpandedId(id)
    setLoadingDetail(true)
    try {
      const detail = await modelApi.get(id)
      setExpandedModel(detail)
    } catch (err) {
      console.error('Failed to load model detail:', err)
    } finally {
      setLoadingDetail(false)
    }
  }

  const handleSetCurrent = async (id: string) => {
    setActionLoading(true)
    try {
      await modelApi.setCurrent(id)
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to set current model')
    } finally {
      setActionLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    setActionLoading(true)
    try {
      await modelApi.delete(id)
      setDeleteConfirm(null)
      if (expandedId === id) {
        setExpandedId(null)
        setExpandedModel(null)
      }
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete model')
    } finally {
      setActionLoading(false)
    }
  }

  // 生成可用的驗證結束月份選項（valid_end，也是模型命名的月份）
  // 從資料結束日期往前推算，以月為單位
  const monthOptions = useCallback(() => {
    if (!dataRange) return []

    const options: { value: string; label: string }[] = []
    const endDate = new Date(dataRange.end)
    const startDate = new Date(dataRange.start)

    // 從資料結束月份開始，往前生成選項
    const current = new Date(endDate.getFullYear(), endDate.getMonth(), 1)

    // 最多往前 24 個月，且不超過資料開始日期
    for (let i = 0; i < 24; i++) {
      if (current < startDate) break

      const value = `${current.getFullYear()}-${String(current.getMonth() + 1).padStart(2, '0')}`
      const label = `${current.getFullYear()}/${String(current.getMonth() + 1).padStart(2, '0')}`
      options.push({ value, label })
      current.setMonth(current.getMonth() - 1)
    }

    return options // 最新的在前
  }, [dataRange])

  const handleStartTraining = async () => {
    if (isTraining) {
      return // 訓練進行中，不允許重複觸發
    }
    setActionLoading(true)
    try {
      // selectedMonth 是 valid_end 月份，計算 train_end = valid_end - 126 天
      const [year, month] = selectedMonth.split('-').map(Number)
      const validEndDate = new Date(year, month, 0) // 該月最後一天
      // train_end = valid_end - VALID_DAYS (126 天)
      const trainEndDate = new Date(validEndDate)
      trainEndDate.setDate(trainEndDate.getDate() - 126)
      const trainEnd = trainEndDate.toISOString().split('T')[0]

      await modelApi.train({
        train_end: trainEnd,
        hyperparams_id: selectedHpId ?? undefined,
      })
      // 訓練已啟動，立即刷新列表
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setActionLoading(false)
    }
  }

  const handleCancelTraining = async () => {
    if (!activeJob) return
    setActionLoading(true)
    try {
      await cancelJob(activeJob.id)
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to cancel training')
    } finally {
      setActionLoading(false)
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('zh-TW', { year: 'numeric', month: '2-digit', day: '2-digit' })
  }

  const formatRelativeTime = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays} days ago`
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
    return `${Math.floor(diffDays / 30)} months ago`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red">{error}</p>
        <button className="btn btn-secondary" onClick={fetchData}>
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    )
  }

  const completedModels = models.filter(m => m.status === 'completed')
  const currentModel = completedModels.find(m => m.is_current)
  const bestIc = completedModels.length > 0
    ? Math.max(...completedModels.map(h => h.metrics.ic || 0))
    : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Models</h1>
          <p className="subheading mt-1">
            Manage trained models and start new training.
            {!isConnected && (
              <span className="text-orange ml-2">(WebSocket disconnected)</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* 驗證結束月份選擇（模型命名月份）*/}
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">Model:</label>
            <select
              className="px-3 py-2 rounded-lg border border-border bg-background text-sm"
              value={selectedMonth}
              onChange={(e) => setSelectedMonth(e.target.value)}
              disabled={isTraining}
            >
              {monthOptions().map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          {/* 超參數選擇 */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">Hyperparams:</label>
            {hyperparamsList.length > 0 ? (
              <select
                className="px-3 py-2 rounded-lg border border-border bg-background text-sm"
                value={selectedHpId ?? ''}
                onChange={(e) => setSelectedHpId(e.target.value ? Number(e.target.value) : null)}
                disabled={isTraining}
              >
                {hyperparamsList.map((hp) => (
                  <option key={hp.id} value={hp.id}>
                    {hp.name}{hp.is_current ? ' (Current)' : ''}
                  </option>
                ))}
              </select>
            ) : (
              <Link
                to="/models/hyperparams"
                className="text-sm text-blue hover:underline"
              >
                Cultivate First
              </Link>
            )}
          </div>
          <button
            className="btn btn-primary"
            onClick={handleStartTraining}
            disabled={actionLoading || isTraining || !selectedMonth || hyperparamsList.length === 0}
          >
            {isTraining ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {isTraining ? 'Training...' : 'New Training'}
          </button>
        </div>
      </div>

      {/* Training Progress */}
      {isTraining && activeJob && (
        <div className="p-4 rounded-lg bg-blue/10 border border-blue/20">
          <div className="flex items-center gap-3 mb-2">
            <Loader2 className="h-5 w-5 animate-spin text-blue" />
            <div className="flex-1">
              <p className="font-semibold text-blue">Training in Progress</p>
              <p className="text-sm text-muted-foreground">
                {activeJob.message || 'Processing...'}
              </p>
            </div>
            <span className="font-mono text-lg text-blue">
              {typeof activeJob.progress === 'number' ? activeJob.progress.toFixed(1) : activeJob.progress}%
            </span>
            <button
              className="btn btn-sm btn-ghost text-red hover:bg-red/10"
              onClick={handleCancelTraining}
              disabled={actionLoading}
              title="Cancel training"
            >
              <AlertTriangle className="h-4 w-4" />
              Cancel
            </button>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-blue transition-all duration-300"
              style={{ width: `${activeJob.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Training Completed */}
      {activeJob?.status === 'completed' && activeJob?.job_type === 'train' && (
        <div className="p-4 rounded-lg bg-green/10 border border-green/20">
          <div className="flex items-center gap-3">
            <CheckCircle className="h-5 w-5 text-green" />
            <div>
              <p className="font-semibold text-green">Training Completed</p>
              <p className="text-sm text-muted-foreground">
                Model trained successfully!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Training Failed */}
      {activeJob?.status === 'failed' && activeJob?.job_type === 'train' && (
        <div className="p-4 rounded-lg bg-red/10 border border-red/20">
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-red" />
            <div>
              <p className="font-semibold text-red">Training Failed</p>
              <p className="text-sm text-muted-foreground">
                {activeJob.error || 'An error occurred during training.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Retrain Warning */}
      {status?.needs_retrain && (
        <div className="flex items-center gap-3 p-4 rounded-lg bg-orange/10 border border-orange/20">
          <AlertTriangle className="h-5 w-5 text-orange" />
          <div>
            <p className="font-semibold text-orange">Model Retrain Recommended</p>
            <p className="text-sm text-muted-foreground">
              Last trained {status.days_since_training} days ago. Consider retraining for better accuracy.
            </p>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Models</span>
            <div className="icon-box icon-box-blue">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{completedModels.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Current IC</span>
            <div className="icon-box icon-box-green">
              <CheckCircle className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${currentModel?.metrics.ic ? 'text-green' : 'text-muted-foreground'}`}>
            {currentModel?.metrics.ic?.toFixed(4) || '---'}
          </p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Best IC</span>
            <div className="icon-box icon-box-purple">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${bestIc ? 'text-purple' : 'text-muted-foreground'}`}>
            {bestIc?.toFixed(4) || '---'}
          </p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Days Since Training</span>
            <div className="icon-box icon-box-orange">
              <Clock className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${status?.needs_retrain ? 'text-orange' : ''}`}>
            {status?.days_since_training ?? '---'}
          </p>
        </div>
      </div>

      {/* Model List */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue" />
            Model List
          </CardTitle>
          <span className="badge badge-gray">{models.length} models</span>
        </CardHeader>
        <CardContent className="p-0">
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <BarChart3 className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Models</p>
              <p className="text-sm text-muted-foreground mt-1">Click "New Training" to create a model</p>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {models.map((model) => (
                <div key={model.id} className="hover:bg-secondary/30">
                  {/* Model Row */}
                  <div
                    className="flex items-center gap-4 px-5 py-4 cursor-pointer"
                    onClick={() => handleExpand(model.id)}
                  >
                    {/* Expand Icon */}
                    <div className="text-muted-foreground">
                      {expandedId === model.id ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </div>

                    {/* Name */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold mono">{model.name || model.id}</span>
                        {model.is_current && (
                          <span className="badge badge-green">Current</span>
                        )}
                        {model.status !== 'completed' && (
                          <span className="badge badge-orange">{model.status}</span>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {model.train_period && (
                          <span>Train: {model.train_period.start} ~ {model.train_period.end}</span>
                        )}
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">IC</p>
                        <p className={`font-semibold mono ${(model.metrics.ic || 0) >= 0.05 ? 'text-green' : ''}`}>
                          {model.metrics.ic?.toFixed(4) || '---'}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">ICIR</p>
                        <p className="mono">{model.metrics.icir?.toFixed(2) || '---'}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">Factors</p>
                        <p>
                          {model.factor_count || 0}
                          {model.candidate_count && (
                            <span className="text-muted-foreground">/{model.candidate_count}</span>
                          )}
                        </p>
                      </div>
                      <div className="text-right w-24">
                        <p className="text-xs text-muted-foreground">Trained</p>
                        <p className="text-sm">{formatRelativeTime(model.trained_at)}</p>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                      {!model.is_current && model.status === 'completed' && (
                        <button
                          className="btn btn-sm btn-ghost"
                          onClick={() => handleSetCurrent(model.id)}
                          disabled={actionLoading}
                          title="Set as current"
                        >
                          <Star className="h-4 w-4" />
                        </button>
                      )}
                      {!model.is_current && (
                        <button
                          className={`btn btn-sm btn-ghost ${['running', 'queued'].includes(model.status) ? 'text-orange hover:bg-orange/10' : 'text-red hover:bg-red/10'}`}
                          onClick={() => setDeleteConfirm(model.id)}
                          disabled={actionLoading}
                          title={['running', 'queued'].includes(model.status) ? 'Cancel training' : 'Delete'}
                        >
                          {['running', 'queued'].includes(model.status) ? (
                            <XCircle className="h-4 w-4" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Expanded Detail */}
                  {expandedId === model.id && (
                    <div className="px-5 pb-4 pt-0">
                      {loadingDetail ? (
                        <div className="flex items-center justify-center py-8">
                          <Loader2 className="h-6 w-6 animate-spin text-primary" />
                        </div>
                      ) : expandedModel ? (
                        <div className="ml-8 space-y-4">
                          {/* Periods */}
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 rounded-lg bg-secondary">
                              <p className="text-xs text-muted-foreground mb-1">Training Period</p>
                              <p className="font-semibold">
                                {expandedModel.train_period
                                  ? `${formatDate(expandedModel.train_period.start)} ~ ${formatDate(expandedModel.train_period.end)}`
                                  : '---'}
                              </p>
                            </div>
                            <div className="p-3 rounded-lg bg-secondary">
                              <p className="text-xs text-muted-foreground mb-1">Validation Period</p>
                              <p className="font-semibold">
                                {expandedModel.valid_period
                                  ? `${formatDate(expandedModel.valid_period.start)} ~ ${formatDate(expandedModel.valid_period.end)}`
                                  : '---'}
                              </p>
                            </div>
                          </div>

                          {/* Selected Factors */}
                          {expandedModel.selected_factors.length > 0 && (
                            <div>
                              <p className="text-xs text-muted-foreground mb-2">
                                Selected Factors ({expandedModel.selected_factors.length})
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {expandedModel.selected_factors.map((f: FactorSummary) => (
                                  <span
                                    key={f.id}
                                    className="badge badge-blue"
                                    title={f.ic_value ? `IC: ${f.ic_value.toFixed(4)}` : undefined}
                                  >
                                    {f.display_name || f.name}
                                    {f.ic_value && (
                                      <span className="ml-1 opacity-70">
                                        ({f.ic_value.toFixed(3)})
                                      </span>
                                    )}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Candidate Factors (if different from selected) */}
                          {expandedModel.candidate_factors.length > expandedModel.selected_factors.length && (
                            <div>
                              <p className="text-xs text-muted-foreground mb-2">
                                Candidate Factors ({expandedModel.candidate_factors.length})
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {expandedModel.candidate_factors.map((f: FactorSummary) => {
                                  const isSelected = expandedModel.selected_factors.some(s => s.id === f.id)
                                  return (
                                    <span
                                      key={f.id}
                                      className={`badge ${isSelected ? 'badge-blue' : 'badge-gray'}`}
                                    >
                                      {f.display_name || f.name}
                                    </span>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      {deleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-2">Delete Model</h3>
            <p className="text-muted-foreground mb-4">
              Are you sure you want to delete this model? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                className="btn btn-secondary"
                onClick={() => setDeleteConfirm(null)}
                disabled={actionLoading}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary bg-red hover:bg-red/90"
                onClick={() => handleDelete(deleteConfirm)}
                disabled={actionLoading}
              >
                {actionLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
