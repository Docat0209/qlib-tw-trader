import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  SlidersHorizontal,
  Loader2,
  RefreshCw,
  Star,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Zap,
  X,
} from 'lucide-react'
import { hyperparamsApi, HyperparamsSummary, HyperparamsDetail, CultivateRequest } from '@/api/client'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { cn } from '@/lib/utils'

export function Hyperparams() {
  const [hyperparams, setHyperparams] = useState<HyperparamsSummary[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<HyperparamsDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDialog, setShowDialog] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null)

  // WebSocket 任務進度追蹤
  const { activeJob, clearJob, cancelJob, isConnected } = useJobs()

  // 培養狀態
  const isCultivating = activeJob?.job_type === 'cultivate' &&
    ['queued', 'running'].includes(activeJob.status)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await hyperparamsApi.list()
      setHyperparams(res.items)
      // 預設選擇 current
      if (!selectedId && res.items.length > 0) {
        const current = res.items.find(hp => hp.is_current)
        setSelectedId(current?.id || res.items[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [selectedId])

  // 載入詳情
  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    setLoadingDetail(true)
    hyperparamsApi.get(selectedId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoadingDetail(false))
  }, [selectedId])

  // 初始載入
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // 自動刷新
  useFetchOnChange('hyperparams', fetchData)

  // 監聽培養完成
  useEffect(() => {
    if (activeJob?.status === 'completed' && activeJob?.job_type === 'cultivate') {
      fetchData()
      const timer = setTimeout(() => clearJob(activeJob.id), 5000)
      return () => clearTimeout(timer)
    }
    if (activeJob?.status === 'failed' && activeJob?.job_type === 'cultivate') {
      fetchData()
    }
  }, [activeJob?.status, activeJob?.job_type, activeJob?.id, clearJob, fetchData])

  const handleSetCurrent = async (id: number) => {
    setActionLoading(true)
    try {
      await hyperparamsApi.setCurrent(id)
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to set current')
    } finally {
      setActionLoading(false)
    }
  }

  const handleDelete = async (id: number) => {
    setActionLoading(true)
    try {
      await hyperparamsApi.delete(id)
      setDeleteConfirm(null)
      if (selectedId === id) {
        setSelectedId(null)
        setDetail(null)
      }
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete')
    } finally {
      setActionLoading(false)
    }
  }

  const handleCancelCultivation = async () => {
    if (!activeJob) return
    setActionLoading(true)
    try {
      await cancelJob(activeJob.id)
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to cancel')
    } finally {
      setActionLoading(false)
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('zh-TW', { year: 'numeric', month: '2-digit', day: '2-digit' })
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Hyperparameters</h1>
          <p className="subheading mt-1">
            Manage LightGBM hyperparameter sets.
            {!isConnected && (
              <span className="text-orange ml-2">(WebSocket disconnected)</span>
            )}
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => setShowDialog(true)}
          disabled={isCultivating}
        >
          {isCultivating ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Zap className="h-4 w-4" />
          )}
          {isCultivating ? 'Cultivating...' : 'Cultivate New'}
        </button>
      </div>

      {/* Cultivation Progress */}
      {isCultivating && activeJob && (
        <div className="p-4 rounded-lg bg-purple/10 border border-purple/20">
          <div className="flex items-center gap-3 mb-2">
            <Loader2 className="h-5 w-5 animate-spin text-purple" />
            <div className="flex-1">
              <p className="font-semibold text-purple">Cultivation in Progress</p>
              <p className="text-sm text-muted-foreground">
                {activeJob.message || 'Processing...'}
              </p>
            </div>
            <span className="font-mono text-lg text-purple">
              {typeof activeJob.progress === 'number' ? activeJob.progress.toFixed(1) : activeJob.progress}%
            </span>
            <button
              className="btn btn-sm btn-ghost text-red hover:bg-red/10"
              onClick={handleCancelCultivation}
              disabled={actionLoading}
              title="Cancel cultivation"
            >
              <AlertTriangle className="h-4 w-4" />
              Cancel
            </button>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-purple transition-all duration-300"
              style={{ width: `${activeJob.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Cultivation Completed */}
      {activeJob?.status === 'completed' && activeJob?.job_type === 'cultivate' && (
        <div className="p-4 rounded-lg bg-green/10 border border-green/20">
          <div className="flex items-center gap-3">
            <CheckCircle className="h-5 w-5 text-green" />
            <div>
              <p className="font-semibold text-green">Cultivation Completed</p>
              <p className="text-sm text-muted-foreground">
                Hyperparameters cultivated successfully!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Cultivation Failed */}
      {activeJob?.status === 'failed' && activeJob?.job_type === 'cultivate' && (
        <div className="p-4 rounded-lg bg-red/10 border border-red/20">
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-red" />
            <div>
              <p className="font-semibold text-red">Cultivation Failed</p>
              <p className="text-sm text-muted-foreground">
                {activeJob.error || 'An error occurred during cultivation.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content: 2 Column Layout */}
      <div className="grid grid-cols-3 gap-6">
        {/* Left: List */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-purple" />
              Hyperparameter Sets
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {hyperparams.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 px-4">
                <div className="icon-box icon-box-purple w-12 h-12 mb-4">
                  <SlidersHorizontal className="h-6 w-6" />
                </div>
                <p className="font-semibold">No Hyperparameters</p>
                <p className="text-sm text-muted-foreground mt-1 text-center">
                  Click "Cultivate New" to create
                </p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {hyperparams.map(hp => (
                  <div
                    key={hp.id}
                    className={cn(
                      "p-4 cursor-pointer transition-colors",
                      selectedId === hp.id ? "bg-primary/10" : "hover:bg-secondary/50"
                    )}
                    onClick={() => setSelectedId(hp.id)}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{hp.name}</span>
                      {hp.is_current && (
                        <span className="badge badge-green">Current</span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {formatDate(hp.cultivated_at)} | {hp.n_periods} periods
                    </div>
                    {hp.learning_rate && hp.num_leaves && (
                      <div className="text-xs text-muted-foreground mt-1 mono">
                        lr={hp.learning_rate.toFixed(3)}, leaves={hp.num_leaves}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right: Detail */}
        <Card className="col-span-2">
          {loadingDetail ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          ) : detail ? (
            <>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>{detail.name}</CardTitle>
                <div className="flex items-center gap-2">
                  {!detail.is_current && (
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={() => handleSetCurrent(detail.id)}
                      disabled={actionLoading}
                    >
                      <Star className="h-4 w-4" />
                      Set Current
                    </button>
                  )}
                  {!detail.is_current && (
                    <button
                      className="btn btn-sm btn-ghost text-red hover:bg-red/10"
                      onClick={() => setDeleteConfirm(detail.id)}
                      disabled={actionLoading}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Meta */}
                <div className="flex gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Cultivated:</span>{' '}
                    <span className="font-medium">{formatDate(detail.cultivated_at)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Periods:</span>{' '}
                    <span className="font-medium">{detail.n_periods}</span>
                  </div>
                </div>

                {/* Parameters Table */}
                <div>
                  <h3 className="text-sm font-semibold mb-3">Parameters</h3>
                  <div className="grid grid-cols-4 gap-3">
                    {Object.entries(detail.params)
                      .filter(([key]) => !['objective', 'metric', 'boosting_type', 'verbosity', 'seed', 'feature_pre_filter'].includes(key))
                      .map(([key, value]) => (
                        <div key={key} className="p-3 rounded-lg bg-secondary">
                          <p className="text-xs text-muted-foreground truncate">{key}</p>
                          <p className="font-mono font-semibold">
                            {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : value}
                          </p>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Stability */}
                <div>
                  <h3 className="text-sm font-semibold mb-3">
                    Stability (CV)
                    <span className="font-normal text-muted-foreground ml-2">
                      CV &lt; 0.3 = stable
                    </span>
                  </h3>
                  <div className="grid grid-cols-4 gap-3">
                    {Object.entries(detail.stability).map(([key, cv]) => (
                      <div key={key} className="text-center p-2 rounded-lg bg-secondary">
                        <p className="text-xs text-muted-foreground truncate">{key}</p>
                        <p className={cn(
                          "font-mono font-semibold",
                          cv < 0.3 ? "text-green" : cv < 0.4 ? "text-orange" : "text-red"
                        )}>
                          {cv.toFixed(3)}
                          {cv < 0.3 && <CheckCircle className="inline h-3 w-3 ml-1" />}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Period Results */}
                <div>
                  <h3 className="text-sm font-semibold mb-3">Period Results</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-muted-foreground border-b border-border">
                          <th className="text-left py-2 font-medium">#</th>
                          <th className="text-left py-2 font-medium">Train Period</th>
                          <th className="text-left py-2 font-medium">Valid Period</th>
                          <th className="text-right py-2 font-medium">IC</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.periods.map((p, i) => (
                          <tr key={i} className="border-b border-border/50">
                            <td className="py-2 text-muted-foreground">{i + 1}</td>
                            <td className="py-2">{p.train_start} ~ {p.train_end}</td>
                            <td className="py-2">{p.valid_start} ~ {p.valid_end}</td>
                            <td className="py-2 text-right font-mono">
                              {p.best_ic.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </CardContent>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-64">
              <SlidersHorizontal className="h-12 w-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">Select a hyperparameter set</p>
            </div>
          )}
        </Card>
      </div>

      {/* Cultivate Dialog */}
      {showDialog && (
        <CultivateDialog
          onClose={() => setShowDialog(false)}
          onSubmit={() => {
            setShowDialog(false)
            fetchData()
          }}
        />
      )}

      {/* Delete Confirmation Dialog */}
      {deleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-2">Delete Hyperparameters</h3>
            <p className="text-muted-foreground mb-4">
              Are you sure you want to delete this hyperparameter set? This action cannot be undone.
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

// Cultivate Dialog Component
function CultivateDialog({
  onClose,
  onSubmit,
}: {
  onClose: () => void
  onSubmit: () => void
}) {
  const [name, setName] = useState('')
  const [nPeriods, setNPeriods] = useState(5)
  const [nTrials, setNTrials] = useState(20)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return

    setSubmitting(true)
    setError(null)
    try {
      await hyperparamsApi.cultivate({
        name: name.trim(),
        n_periods: nPeriods,
        n_trials_per_period: nTrials,
      })
      onSubmit()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start cultivation')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-card p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Cultivate Hyperparameters</h3>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name *</label>
            <input
              type="text"
              className="input w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Default v1"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Periods (3-10)</label>
            <input
              type="number"
              className="input w-full"
              value={nPeriods}
              onChange={(e) => setNPeriods(Number(e.target.value))}
              min={3}
              max={10}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Number of walk-forward windows
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Trials per Period (10-50)</label>
            <input
              type="number"
              className="input w-full"
              value={nTrials}
              onChange={(e) => setNTrials(Number(e.target.value))}
              min={10}
              max={50}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Optuna trials per window
            </p>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red/10 border border-red/20 text-red text-sm">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || !name.trim()}
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Zap className="h-4 w-4" />
              )}
              Start Cultivation
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
