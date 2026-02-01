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
  Trash2,
  XCircle,
} from 'lucide-react'
import { modelApi, ModelSummary, ModelStatus, Model, FactorSummary, DataRangeResponse, hyperparamsApi, HyperparamsSummary } from '@/api/client'
import { Link } from 'react-router-dom'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { cn } from '@/lib/utils'

export function Training() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [dataRange, setDataRange] = useState<DataRangeResponse | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<string>('')
  const [hyperparamsList, setHyperparamsList] = useState<HyperparamsSummary[]>([])
  const [selectedHpId, setSelectedHpId] = useState<number | null>(null)

  // WebSocket 訓練進度追蹤
  const { activeJob, clearJob, cancelJob } = useJobs()

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
      // 載入超參數列表並預設選擇第一個
      setHyperparamsList(hpRes.items)
      if (selectedHpId === null && hpRes.items.length > 0) {
        setSelectedHpId(hpRes.items[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [selectedMonth, selectedHpId])

  // 載入選中模型詳情
  useEffect(() => {
    if (!selectedId) {
      setSelectedModel(null)
      return
    }
    setLoadingDetail(true)
    modelApi.get(selectedId)
      .then(setSelectedModel)
      .catch(() => setSelectedModel(null))
      .finally(() => setLoadingDetail(false))
  }, [selectedId])

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

  const handleDelete = async (id: string) => {
    setActionLoading(true)
    try {
      await modelApi.delete(id)
      setDeleteConfirm(null)
      if (selectedId === id) {
        setSelectedId(null)
        setSelectedModel(null)
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
  const currentModel = completedModels[0]  // 按名稱降序，第一個是最新的
  const bestIc = completedModels.length > 0
    ? Math.max(...completedModels.map(h => h.metrics.ic || 0))
    : null

  return (
    <div className="flex gap-6 h-[calc(100vh-100px)]">
      {/* 左側：Model List */}
      <div className="w-72 shrink-0 flex flex-col">
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardHeader className="shrink-0 flex flex-row items-center justify-between py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Layers className="h-4 w-4 text-blue" />
              Models
            </CardTitle>
            <div className="flex items-center gap-2">
              <span className="badge badge-gray text-xs">{models.length}</span>
              <button onClick={fetchData} className="p-1 hover:bg-secondary rounded">
                <RefreshCw className="h-3 w-3" />
              </button>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-0 overflow-y-auto">
            {models.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <BarChart3 className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm">No models yet</p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {models.map((model) => (
                  <div
                    key={model.id}
                    onClick={() => setSelectedId(model.id)}
                    className={cn(
                      "w-full text-left p-3 hover:bg-secondary/50 transition-colors cursor-pointer group",
                      selectedId === model.id && "bg-blue-50 border-l-2 border-blue"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold mono text-sm">{model.name || model.id}</span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setDeleteConfirm(model.id)
                          }}
                          className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red/10 transition-opacity"
                          title="Delete"
                        >
                          <Trash2 className="h-3 w-3 text-red" />
                        </button>
                        {model.status === 'completed' ? (
                          <CheckCircle className="h-3 w-3 text-green" />
                        ) : model.status === 'running' ? (
                          <Loader2 className="h-3 w-3 animate-spin text-blue" />
                        ) : model.status === 'failed' ? (
                          <XCircle className="h-3 w-3 text-red" />
                        ) : (
                          <Clock className="h-3 w-3 text-gray-400" />
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground mb-1">
                      {model.train_period && (
                        <span>{model.train_period.start?.slice(5)} ~ {model.train_period.end?.slice(5)}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className={cn(
                        "font-semibold",
                        (model.metrics.ic || 0) >= 0.05 ? 'text-green' : ''
                      )}>
                        IC: {model.metrics.ic?.toFixed(4) || '---'}
                      </span>
                      <span className="text-muted-foreground">
                        {model.factor_count || 0} factors
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 右側：主內容區 */}
      <div className="flex-1 flex flex-col gap-4 overflow-y-auto pr-1">
        {/* Training Progress */}
        {isTraining && activeJob && (
          <div className="p-4 rounded-lg bg-blue/10 border border-blue/20 shrink-0">
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
          <div className="p-4 rounded-lg bg-green/10 border border-green/20 shrink-0">
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
          <div className="p-4 rounded-lg bg-red/10 border border-red/20 shrink-0">
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
          <div className="flex items-center gap-3 p-4 rounded-lg bg-orange/10 border border-orange/20 shrink-0">
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
        <div className="grid grid-cols-4 gap-4 shrink-0">
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

        {/* Row 2: Training Form + Model Detail */}
        <div className="grid grid-cols-2 gap-4 shrink-0">
          {/* Training Form */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Play className="h-4 w-4 text-green" />
                New Training
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 pt-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Model Month</label>
                  <select
                    className="input w-full text-sm"
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
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Hyperparams</label>
                  {hyperparamsList.length > 0 ? (
                    <select
                      className="input w-full text-sm"
                      value={selectedHpId ?? ''}
                      onChange={(e) => setSelectedHpId(e.target.value ? Number(e.target.value) : null)}
                      disabled={isTraining}
                    >
                      {hyperparamsList.map((hp) => (
                        <option key={hp.id} value={hp.id}>
                          {hp.name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <Link
                      to="/models/hyperparams"
                      className="input w-full text-sm flex items-center justify-center text-blue hover:underline"
                    >
                      Cultivate First
                    </Link>
                  )}
                </div>
              </div>
              <button
                className="btn btn-primary w-full text-sm disabled:opacity-50"
                onClick={handleStartTraining}
                disabled={actionLoading || isTraining || !selectedMonth || hyperparamsList.length === 0}
              >
                {isTraining ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isTraining ? 'Training...' : 'Start Training'}
              </button>
            </CardContent>
          </Card>

          {/* Model Detail */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-blue" />
                {selectedModel ? selectedModel.name || selectedModel.id : 'Model Detail'}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {loadingDetail ? (
                <div className="flex items-center justify-center h-24">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              ) : selectedModel ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">IC</p>
                      <p className={`font-semibold text-sm ${(selectedModel.metrics.ic || 0) >= 0.05 ? 'text-green' : ''}`}>
                        {selectedModel.metrics.ic?.toFixed(4) || '---'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">ICIR</p>
                      <p className="font-semibold text-sm">
                        {selectedModel.metrics.icir?.toFixed(2) || '---'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Factors</p>
                      <p className="font-semibold text-sm">
                        {selectedModel.factor_count || 0}
                        {selectedModel.candidate_factors?.length > 0 && (
                          <span className="text-muted-foreground font-normal">/{selectedModel.candidate_factors.length}</span>
                        )}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Trained</p>
                      <p className="font-semibold text-sm">
                        {formatRelativeTime(selectedModel.trained_at)}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p>Train: {selectedModel.train_period ? `${formatDate(selectedModel.train_period.start)} ~ ${formatDate(selectedModel.train_period.end)}` : '---'}</p>
                    <p>Valid: {selectedModel.valid_period ? `${formatDate(selectedModel.valid_period.start)} ~ ${formatDate(selectedModel.valid_period.end)}` : '---'}</p>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
                  Select a model to view details
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Factor Pool & Selected Factors */}
        {selectedModel && selectedModel.candidate_factors.length > 0 && (
          <Card className="shrink-0">
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Layers className="h-4 w-4 text-orange" />
                Factor Pool
                <span className="text-muted-foreground font-normal text-sm">
                  {selectedModel.selected_factors.length} / {selectedModel.candidate_factors.length} selected
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              {/* Candidate Factor Pool */}
              <div>
                <p className="text-xs text-muted-foreground mb-2">
                  Candidate Factors ({selectedModel.candidate_factors.length})
                </p>
                <div className="flex flex-wrap gap-2">
                  {selectedModel.candidate_factors.map((f: FactorSummary) => {
                    const isSelected = selectedModel.selected_factors.some(sf => sf.id === f.id)
                    return (
                      <span
                        key={f.id}
                        className={cn(
                          "badge",
                          isSelected ? "badge-blue" : "badge-gray"
                        )}
                        title={f.ic_value ? `IC: ${f.ic_value.toFixed(4)}` : undefined}
                      >
                        {f.display_name || f.name}
                      </span>
                    )
                  })}
                </div>
              </div>

              {/* Selected Factors with IC */}
              {selectedModel.selected_factors.length > 0 && (
                <div className="pt-3 border-t border-border">
                  <p className="text-xs text-muted-foreground mb-2">
                    Selected Factors ({selectedModel.selected_factors.length}) - IC Incremental
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.selected_factors.map((f: FactorSummary, idx: number) => (
                      <span
                        key={f.id}
                        className="badge badge-purple"
                        title={f.ic_value ? `IC: ${f.ic_value.toFixed(4)}` : undefined}
                      >
                        <span className="text-[10px] opacity-60 mr-1">#{idx + 1}</span>
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
            </CardContent>
          </Card>
        )}
      </div>

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
