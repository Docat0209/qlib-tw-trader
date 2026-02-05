import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  SlidersHorizontal,
  Loader2,
  RefreshCw,
  Trash2,
  CheckCircle,
} from 'lucide-react'
import { hyperparamsApi, HyperparamsSummary, HyperparamsDetail } from '@/api/client'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { cn } from '@/lib/utils'

export function Hyperparams() {
  const [hyperparams, setHyperparams] = useState<HyperparamsSummary[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<HyperparamsDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await hyperparamsApi.list()
      setHyperparams(res.items)
      // 預設選擇第一個
      if (!selectedId && res.items.length > 0) {
        setSelectedId(res.items[0].id)
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
            View LightGBM hyperparameter sets. Select one when running walk-forward backtest.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={fetchData}
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Main Content: Left-Right Layout */}
      <div className="flex gap-6 h-[calc(100vh-200px)]">
        {/* Left: List */}
        <div className="w-72 shrink-0 flex flex-col">
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-purple" />
              Hyperparameter Sets
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 overflow-y-auto">
            {hyperparams.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 px-4">
                <div className="icon-box icon-box-purple w-12 h-12 mb-4">
                  <SlidersHorizontal className="h-6 w-6" />
                </div>
                <p className="font-semibold">No Hyperparameters</p>
                <p className="text-sm text-muted-foreground mt-1 text-center">
                  Add hyperparameter sets via database
                </p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {hyperparams.map(hp => (
                  <div
                    key={hp.id}
                    className={cn(
                      "p-4 cursor-pointer transition-colors group",
                      selectedId === hp.id ? "bg-primary/10" : "hover:bg-secondary/50"
                    )}
                    onClick={() => setSelectedId(hp.id)}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{hp.name}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setDeleteConfirm(hp.id)
                        }}
                        className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red/10 transition-opacity"
                        title="Delete"
                      >
                        <Trash2 className="h-3 w-3 text-red" />
                      </button>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {formatDate(hp.cultivated_at)}
                    </div>
                    {hp.learning_rate && hp.num_leaves && (
                      <div className="text-xs text-muted-foreground mt-1 mono">
                        lr={hp.learning_rate.toFixed(3)}, leaves={hp.num_leaves}
                      </div>
                    )}
                    {hp.lambda_l1 && hp.lambda_l2 && (
                      <div className="text-xs text-muted-foreground mono">
                        L1={hp.lambda_l1.toFixed(1)}, L2={hp.lambda_l2.toFixed(1)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
        </div>

        {/* Right: Detail */}
        <Card className="flex-1 overflow-y-auto">
          {loadingDetail ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          ) : detail ? (
            <>
              <CardHeader>
                <CardTitle>{detail.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Meta */}
                <div className="flex gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Created:</span>{' '}
                    <span className="font-medium">{formatDate(detail.cultivated_at)}</span>
                  </div>
                  {detail.n_periods > 0 && (
                    <div>
                      <span className="text-muted-foreground">Periods:</span>{' '}
                      <span className="font-medium">{detail.n_periods}</span>
                    </div>
                  )}
                </div>

                {/* Source Info */}
                {detail.stability.source && (
                  <div className="p-3 rounded-lg bg-purple/10 border border-purple/20">
                    <p className="text-sm">
                      <span className="text-muted-foreground">Source:</span>{' '}
                      <span className="font-medium text-purple">{detail.stability.source}</span>
                    </p>
                    {detail.stability.reference && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Reference: {detail.stability.reference}
                      </p>
                    )}
                    {detail.stability.factor_ratio && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Factor ratio: {detail.stability.factor_ratio}x | Stock ratio: {detail.stability.stock_ratio}x
                      </p>
                    )}
                  </div>
                )}

                {/* Parameters Table */}
                <div>
                  <h3 className="text-sm font-semibold mb-3">Parameters</h3>
                  <div className="grid grid-cols-4 gap-3">
                    {Object.entries(detail.params)
                      .filter(([key]) => !['objective', 'metric', 'boosting_type', 'verbosity', 'seed', 'feature_pre_filter', 'device', 'gpu_use_dp'].includes(key))
                      .map(([key, value]) => (
                        <div key={key} className="p-3 rounded-lg bg-secondary">
                          <p className="text-xs text-muted-foreground truncate">{key}</p>
                          <p className="font-mono font-semibold">
                            {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : String(value)}
                          </p>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Stability (only show if it has numeric CV values) */}
                {Object.entries(detail.stability).some(([_, v]) => typeof v === 'number') && (
                  <div>
                    <h3 className="text-sm font-semibold mb-3">
                      Stability (CV)
                      <span className="font-normal text-muted-foreground ml-2">
                        CV &lt; 0.3 = stable
                      </span>
                    </h3>
                    <div className="grid grid-cols-4 gap-3">
                      {Object.entries(detail.stability)
                        .filter(([_, v]) => typeof v === 'number')
                        .map(([key, cv]) => (
                          <div key={key} className="text-center p-2 rounded-lg bg-secondary">
                            <p className="text-xs text-muted-foreground truncate">{key}</p>
                            <p className={cn(
                              "font-mono font-semibold",
                              (cv as number) < 0.3 ? "text-green" : (cv as number) < 0.4 ? "text-orange" : "text-red"
                            )}>
                              {(cv as number).toFixed(3)}
                              {(cv as number) < 0.3 && <CheckCircle className="inline h-3 w-3 ml-1" />}
                            </p>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Period Results (only show if has periods) */}
                {detail.periods.length > 0 && (
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
                )}
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
