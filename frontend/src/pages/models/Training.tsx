import { useEffect, useState } from 'react'
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
} from 'lucide-react'
import { modelApi, ModelSummary, ModelStatus, Model, FactorSummary } from '@/api/client'

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

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [listRes, statusRes] = await Promise.all([
        modelApi.list(),
        modelApi.status(),
      ])
      setModels(listRes.items)
      setStatus(statusRes)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

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

  const handleStartTraining = async () => {
    setActionLoading(true)
    try {
      const res = await modelApi.train()
      alert(res.message)
      await fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start training')
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
          <p className="subheading mt-1">Manage trained models and start new training.</p>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleStartTraining}
          disabled={actionLoading}
        >
          {actionLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          New Training
        </button>
      </div>

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
                          className="btn btn-sm btn-ghost text-red hover:bg-red/10"
                          onClick={() => setDeleteConfirm(model.id)}
                          disabled={actionLoading}
                          title="Delete"
                        >
                          <Trash2 className="h-4 w-4" />
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
