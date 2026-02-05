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
  Calendar,
  SlidersHorizontal,
} from 'lucide-react'
import { modelApi, WeeksResponse, Model, FactorSummary, hyperparamsApi, HyperparamsSummary } from '@/api/client'
import { Link } from 'react-router-dom'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { cn } from '@/lib/utils'
import { WeekCalendar } from '@/components/WeekCalendar'

export function Training() {
  const [weeksData, setWeeksData] = useState<WeeksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedWeekId, setSelectedWeekId] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [, setLoadingDetail] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [hyperparamsList, setHyperparamsList] = useState<HyperparamsSummary[]>([])
  const [selectedHyperparamsId, setSelectedHyperparamsId] = useState<number | null>(null)
  // 刪除所有模型
  const [deleteAllConfirm, setDeleteAllConfirm] = useState(false)
  const [deletingModels, setDeletingModels] = useState(false)
  // 批量訓練年份標記
  const [queueYear, setQueueYear] = useState<string | null>(null)

  // WebSocket 訓練進度追蹤
  const { activeJob, clearJob, cancelJob } = useJobs()

  // 訓練狀態（支援單週和批量訓練）
  const isTraining = (activeJob?.job_type === 'train' || activeJob?.job_type === 'train_batch') &&
    ['queued', 'running'].includes(activeJob.status)

  // 從批量訓練訊息解析剩餘數量 [1/5] Training... => 4
  const getBatchRemaining = useCallback(() => {
    if (activeJob?.job_type !== 'train_batch' || !activeJob.message) return 0
    const match = activeJob.message.match(/\[(\d+)\/(\d+)\]/)
    if (match) {
      const current = parseInt(match[1], 10)
      const total = parseInt(match[2], 10)
      return total - current
    }
    return 0
  }, [activeJob])

  const fetchData = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true)
    setError(null)
    try {
      const [weeksRes, hpRes] = await Promise.all([
        modelApi.weeks(),
        hyperparamsApi.list().catch(() => ({ items: [] })),
      ])
      setWeeksData(weeksRes)
      setHyperparamsList(hpRes.items)
      // 預設選擇第一個超參數
      if (hpRes.items.length > 0 && !selectedHyperparamsId) {
        setSelectedHyperparamsId(hpRes.items[0].id)
      }

      // 預設選擇最新可訓練的週（僅首次載入）
      if (showLoading && weeksRes.slots.length > 0) {
        setSelectedWeekId(prev => {
          if (prev) return prev // 保持現有選擇
          const trainable = weeksRes.slots.find(s => s.status === 'trainable')
          return trainable ? trainable.week_id : weeksRes.slots[0].week_id
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      if (showLoading) setLoading(false)
    }
  }, [selectedHyperparamsId])

  // 選中週變更時，載入對應模型
  useEffect(() => {
    if (!selectedWeekId || !weeksData) {
      setSelectedModel(null)
      return
    }
    const slot = weeksData.slots.find(s => s.week_id === selectedWeekId)
    if (!slot?.model) {
      setSelectedModel(null)
      return
    }
    setLoadingDetail(true)
    modelApi.get(slot.model.id)
      .then(setSelectedModel)
      .catch(() => setSelectedModel(null))
      .finally(() => setLoadingDetail(false))
  }, [selectedWeekId, weeksData])

  // 初始載入
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // 自動刷新（靜默刷新，不顯示 loading）
  const silentRefresh = useCallback(() => fetchData(false), [fetchData])
  useFetchOnChange('models', silentRefresh)
  useFetchOnChange('hyperparams', silentRefresh)

  // 監聽訓練完成
  useEffect(() => {
    const jobType = activeJob?.job_type
    const isTrainJob = jobType === 'train' || jobType === 'train_batch'

    if (activeJob?.status === 'completed' && isTrainJob) {
      fetchData(false)
      setQueueYear(null)

      const timer = setTimeout(() => clearJob(activeJob.id), 3000)
      return () => clearTimeout(timer)
    }
    if (activeJob?.status === 'failed' && isTrainJob) {
      fetchData(false)
      setQueueYear(null)
    }
  }, [activeJob?.status, activeJob?.job_type, activeJob?.id, clearJob, fetchData])

  const selectedSlot = weeksData?.slots.find(s => s.week_id === selectedWeekId)

  const handleDelete = async () => {
    if (!selectedSlot?.model) return
    setActionLoading(true)
    try {
      await modelApi.delete(selectedSlot.model.id)
      setDeleteConfirm(null)
      setSelectedModel(null)
      await fetchData(false)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete model')
    } finally {
      setActionLoading(false)
    }
  }

  const handleStartTraining = async () => {
    if (isTraining || !selectedWeekId || !selectedHyperparamsId) return
    setActionLoading(true)
    try {
      await modelApi.train({ week_id: selectedWeekId, hyperparams_id: selectedHyperparamsId })
      await fetchData(false)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setActionLoading(false)
    }
  }

  const handleTrainYear = async (year: string) => {
    if (isTraining || !weeksData || !selectedHyperparamsId) return

    // 找出該年所有可訓練的週（未訓練）
    const trainableWeeks = weeksData.slots.filter(
      s => s.week_id.startsWith(year) && s.status === 'trainable'
    )
    if (trainableWeeks.length === 0) {
      alert(`${year} 年沒有待訓練的週`)
      return
    }

    setQueueYear(year)

    try {
      // 使用後端批量訓練 API
      await modelApi.trainBatch({ year, hyperparams_id: selectedHyperparamsId })
      await fetchData(false)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start batch training')
      setQueueYear(null)
    }
  }

  const handleCancelTraining = async () => {
    if (!activeJob) return
    setActionLoading(true)
    try {
      await cancelJob(activeJob.id)
      await fetchData(false)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to cancel training')
    } finally {
      setActionLoading(false)
    }
  }

  const handleDeleteAllModels = async () => {
    setDeletingModels(true)
    try {
      const result = await modelApi.deleteAll()
      setDeleteAllConfirm(false)
      alert(`Deleted ${result.deleted_count} models`)
      fetchData(false)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete models')
    } finally {
      setDeletingModels(false)
    }
  }

  const selectedHyperparams = hyperparamsList.find(hp => hp.id === selectedHyperparamsId)

  const formatDate = (dateStr: string) => {
    return dateStr.slice(5) // MM-DD
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
        <button className="btn btn-secondary" onClick={() => fetchData()}>
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    )
  }

  const trainedSlots = weeksData?.slots.filter(s => s.status === 'trained') || []
  const trainableSlots = weeksData?.slots.filter(s => s.status === 'trainable') || []
  const bestIc = trainedSlots.length > 0
    ? Math.max(...trainedSlots.map(s => s.model?.model_ic || 0))
    : null
  const avgIc = trainedSlots.length > 0
    ? trainedSlots.reduce((sum, s) => sum + (s.model?.model_ic || 0), 0) / trainedSlots.length
    : null

  return (
    <div className="flex gap-6 h-[calc(100vh-100px)]">
      {/* 左側：超參數 + 週曆 */}
      <div className="w-80 shrink-0 flex flex-col gap-4">
        {/* Hyperparameters Selection */}
        <Card className="shrink-0">
          <CardHeader className="py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <SlidersHorizontal className="h-4 w-4 text-purple" />
              Hyperparameters
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0 pb-3">
            <select
              className="input w-full text-sm"
              value={selectedHyperparamsId || ''}
              onChange={(e) => setSelectedHyperparamsId(e.target.value ? Number(e.target.value) : null)}
              disabled={hyperparamsList.length === 0}
            >
              {hyperparamsList.length === 0 ? (
                <option value="">No hyperparameters available</option>
              ) : (
                hyperparamsList.map((hp) => (
                  <option key={hp.id} value={hp.id}>
                    {hp.name}
                  </option>
                ))
              )}
            </select>
            {selectedHyperparams && (
              <p className="text-[10px] text-muted-foreground mt-2 font-mono">
                leaves={selectedHyperparams.num_leaves}, L1={selectedHyperparams.lambda_l1?.toFixed(0)}, L2={selectedHyperparams.lambda_l2?.toFixed(0)}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Week Calendar */}
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardHeader className="shrink-0 flex flex-row items-center justify-between py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Calendar className="h-4 w-4 text-blue" />
              Week Calendar
            </CardTitle>
            <div className="flex items-center gap-2">
              <span className="badge badge-green text-xs">{trainedSlots.length} trained</span>
              <span className="badge badge-gray text-xs">{trainableSlots.length} pending</span>
              <button onClick={() => fetchData()} className="p-1 hover:bg-secondary rounded">
                <RefreshCw className="h-3 w-3" />
              </button>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-3 overflow-y-auto">
            {weeksData && weeksData.slots.length > 0 ? (
              <WeekCalendar
                slots={weeksData.slots}
                selected={selectedWeekId}
                onSelect={setSelectedWeekId}
                currentFactorPoolHash={weeksData.current_factor_pool_hash}
                onTrainYear={handleTrainYear}
                isTraining={isTraining || actionLoading}
                queueYear={activeJob?.job_type === 'train_batch' && isTraining ? queueYear : null}
                queueRemaining={getBatchRemaining()}
                onCancelQueue={handleCancelTraining}
              />
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <Calendar className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm">No trainable weeks</p>
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

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 shrink-0">
          <div className="stat-card">
            <div className="flex items-center justify-between">
              <span className="stat-label">Trained Weeks</span>
              <div className="icon-box icon-box-green">
                <Layers className="h-4 w-4" />
              </div>
            </div>
            <p className="stat-value">{trainedSlots.length}</p>
          </div>
          <div className="stat-card">
            <div className="flex items-center justify-between">
              <span className="stat-label">Pending Weeks</span>
              <div className="icon-box icon-box-blue">
                <Clock className="h-4 w-4" />
              </div>
            </div>
            <p className="stat-value">{trainableSlots.length}</p>
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
              <span className="stat-label">Avg IC</span>
              <div className="icon-box icon-box-orange">
                <BarChart3 className="h-4 w-4" />
              </div>
            </div>
            <p className={`stat-value ${avgIc ? 'text-orange' : 'text-muted-foreground'}`}>
              {avgIc?.toFixed(4) || '---'}
            </p>
          </div>
        </div>

        {/* Row 2: Week Detail + Training Action */}
        <div className="grid grid-cols-2 gap-4 shrink-0">
          {/* Week Detail */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Calendar className="h-4 w-4 text-blue" />
                {selectedWeekId || 'Select a Week'}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {selectedSlot ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Status</p>
                      <p className={cn(
                        "font-semibold text-sm",
                        selectedSlot.status === 'trained' && 'text-green',
                        selectedSlot.status === 'trainable' && 'text-blue',
                        selectedSlot.status === 'insufficient_data' && 'text-gray-400'
                      )}>
                        {selectedSlot.status === 'trained' ? 'Trained' :
                         selectedSlot.status === 'trainable' ? 'Trainable' : 'No Data'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Model IC</p>
                      <p className={cn(
                        "font-semibold text-sm",
                        (selectedSlot.model?.model_ic || 0) >= 0.05 && 'text-green'
                      )}>
                        {selectedSlot.model?.model_ic?.toFixed(4) || '---'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Factors</p>
                      <p className="font-semibold text-sm">
                        {selectedSlot.model?.factor_count || '---'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-secondary/50">
                      <p className="text-[10px] text-muted-foreground">Factor Pool</p>
                      <p className={cn(
                        "font-semibold text-sm font-mono",
                        selectedSlot.model?.is_outdated && 'text-yellow-600'
                      )}>
                        {selectedSlot.model?.factor_pool_hash || '---'}
                        {selectedSlot.model?.is_outdated && ' (outdated)'}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Train: {formatDate(selectedSlot.train_start)} ~ {formatDate(selectedSlot.train_end)}</p>
                    <p>Valid: {formatDate(selectedSlot.valid_start)} ~ {formatDate(selectedSlot.valid_end)}</p>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
                  Select a week from the calendar
                </div>
              )}
            </CardContent>
          </Card>

          {/* Training Action */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center justify-between text-base">
                <div className="flex items-center gap-2">
                  <Play className="h-4 w-4 text-green" />
                  Training
                </div>
                <button
                  onClick={() => setDeleteAllConfirm(true)}
                  className="btn btn-sm btn-ghost text-red hover:bg-red/10"
                  title="Delete all models"
                  disabled={trainedSlots.length === 0}
                >
                  <Trash2 className="h-3 w-3" />
                  Clear All
                </button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 pt-4">
              {hyperparamsList.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground mb-2">
                    No hyperparameters available
                  </p>
                  <Link
                    to="/models/hyperparams"
                    className="btn btn-secondary text-sm"
                  >
                    View Hyperparams
                  </Link>
                </div>
              ) : selectedSlot?.status === 'trained' ? (
                <div className="space-y-3">
                  <div className="p-3 rounded bg-green-50 border border-green-200">
                    <div className="flex items-center gap-2 text-green-700">
                      <CheckCircle className="h-4 w-4" />
                      <span className="font-medium">Already Trained</span>
                    </div>
                    <p className="text-xs text-green-600 mt-1">
                      Model: {selectedSlot.model?.name}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      className="btn btn-secondary flex-1 text-sm"
                      onClick={handleStartTraining}
                      disabled={actionLoading || isTraining}
                    >
                      <RefreshCw className="h-4 w-4" />
                      Retrain
                    </button>
                    <button
                      className="btn btn-ghost text-red text-sm"
                      onClick={() => setDeleteConfirm(selectedSlot.model?.id || null)}
                      disabled={actionLoading}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ) : selectedSlot?.status === 'trainable' ? (
                <div className="space-y-3">
                  <div className="p-3 rounded bg-blue-50 border border-blue-200">
                    <div className="flex items-center gap-2 text-blue-700">
                      <Clock className="h-4 w-4" />
                      <span className="font-medium">Ready to Train</span>
                    </div>
                    <p className="text-xs text-blue-600 mt-1">
                      Using: {selectedHyperparams?.name || 'No hyperparams selected'}
                    </p>
                  </div>
                  <button
                    className="btn btn-primary w-full text-sm disabled:opacity-50"
                    onClick={handleStartTraining}
                    disabled={actionLoading || isTraining}
                  >
                    {isTraining ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    {isTraining ? 'Training...' : 'Start Training'}
                  </button>
                </div>
              ) : (
                <div className="p-3 rounded bg-gray-50 border border-gray-200">
                  <div className="flex items-center gap-2 text-gray-500">
                    <XCircle className="h-4 w-4" />
                    <span className="font-medium">Insufficient Data</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    Not enough data to train this week
                  </p>
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

        {/* Data Range Info */}
        {weeksData && (
          <div className="text-xs text-muted-foreground shrink-0">
            Data range: {weeksData.data_range.start} ~ {weeksData.data_range.end}
            {' | '}Factor pool hash: <span className="font-mono">{weeksData.current_factor_pool_hash}</span>
          </div>
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
                onClick={handleDelete}
                disabled={actionLoading}
              >
                {actionLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete All Models Confirmation Dialog */}
      {deleteAllConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red" />
              Delete All Models
            </h3>
            <p className="text-muted-foreground mb-4">
              Are you sure you want to delete <strong>ALL {trainedSlots.length}</strong> trained models?
              This will remove all model files and you will need to retrain from scratch.
            </p>
            <div className="flex justify-end gap-2">
              <button
                className="btn btn-secondary"
                onClick={() => setDeleteAllConfirm(false)}
                disabled={deletingModels}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary bg-red hover:bg-red/90"
                onClick={handleDeleteAllModels}
                disabled={deletingModels}
              >
                {deletingModels ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                Delete All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
