import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Play, Clock, BarChart3, Layers, CheckCircle, Activity, Loader2, RefreshCw, AlertTriangle } from 'lucide-react'
import { modelApi, ModelHistoryItem, ModelStatus, Model } from '@/api/client'

export function Training() {
  const [current, setCurrent] = useState<Model | null>(null)
  const [history, setHistory] = useState<ModelHistoryItem[]>([])
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [historyRes, statusRes] = await Promise.all([
        modelApi.history(20),
        modelApi.status(),
      ])
      setHistory(historyRes.items)
      setStatus(statusRes)

      // 嘗試取得當前模型
      try {
        const currentRes = await modelApi.current()
        setCurrent(currentRes)
      } catch {
        setCurrent(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

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

  const bestIc = history.length > 0
    ? Math.max(...history.map(h => h.metrics.ic || 0))
    : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Training Records</h1>
          <p className="subheading mt-1">View and manage model training history.</p>
        </div>
        <button className="btn btn-primary">
          <Play className="h-4 w-4" />
          Start Training
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
            <span className="stat-label">Total Runs</span>
            <div className="icon-box icon-box-blue">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{history.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Current IC</span>
            <div className="icon-box icon-box-green">
              <CheckCircle className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${current?.metrics.ic ? 'text-green' : 'text-muted-foreground'}`}>
            {current?.metrics.ic?.toFixed(4) || '---'}
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

      {/* Current Model */}
      {current && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green" />
              Current Model
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground mb-1">Model ID</p>
                <p className="font-semibold mono">{current.id}</p>
              </div>
              <div className="p-4 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground mb-1">Factor Count</p>
                <p className="font-semibold">{current.factor_count || '---'}</p>
              </div>
              <div className="p-4 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground mb-1">IC</p>
                <p className="font-semibold text-green mono">{current.metrics.ic?.toFixed(4) || '---'}</p>
              </div>
              <div className="p-4 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground mb-1">ICIR</p>
                <p className="font-semibold mono">{current.metrics.icir?.toFixed(2) || '---'}</p>
              </div>
            </div>
            {current.factors.length > 0 && (
              <div className="mt-4">
                <p className="text-xs text-muted-foreground mb-2">Selected Factors</p>
                <div className="flex flex-wrap gap-2">
                  {current.factors.map(f => (
                    <span key={f} className="badge badge-blue">{f}</span>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Training History */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue" />
            Training History
          </CardTitle>
          <span className="badge badge-gray">{history.length} records</span>
        </CardHeader>
        <CardContent className="p-0">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <BarChart3 className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Training Records</p>
              <p className="text-sm text-muted-foreground mt-1">Click "Start Training" to begin</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">ID</th>
                    <th className="table-header px-5 py-3 text-left">Trained At</th>
                    <th className="table-header px-5 py-3 text-left">Train Period</th>
                    <th className="table-header px-5 py-3 text-right">IC</th>
                    <th className="table-header px-5 py-3 text-right">ICIR</th>
                    <th className="table-header px-5 py-3 text-right">Factors</th>
                    <th className="table-header px-5 py-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((run) => (
                    <tr key={run.id} className="table-row">
                      <td className="table-cell px-5">
                        <span className="font-semibold mono">{run.id}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="mono text-sm">{formatDate(run.trained_at)}</span>
                      </td>
                      <td className="table-cell px-5">
                        {run.train_period ? (
                          <span className="text-sm">
                            {run.train_period.start} ~ {run.train_period.end}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">---</span>
                        )}
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className={`mono font-semibold ${(run.metrics.ic || 0) >= 0.05 ? 'text-green' : ''}`}>
                          {run.metrics.ic?.toFixed(4) || '---'}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="mono">{run.metrics.icir?.toFixed(2) || '---'}</span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span>{run.factor_count || '---'}</span>
                      </td>
                      <td className="table-cell px-5 text-center">
                        {run.is_current ? (
                          <span className="badge badge-green">Current</span>
                        ) : (
                          <span className="badge badge-gray">Archived</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
