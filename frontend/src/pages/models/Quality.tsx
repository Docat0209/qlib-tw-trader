import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  TrendingUp,
  Layers,
  BarChart3,
} from 'lucide-react'
import { modelApi, QualityResponse, QualityMetricsItem } from '@/api/client'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { cn } from '@/lib/utils'

export function Quality() {
  const [data, setData] = useState<QualityResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true)
    setError(null)
    try {
      const res = await modelApi.quality(20)
      setData(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      if (showLoading) setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // 自動刷新
  const silentRefresh = useCallback(() => fetchData(false), [fetchData])
  useFetchOnChange('models', silentRefresh)

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

  const items = data?.items || []
  const thresholds = data?.thresholds

  // 統計
  const warningCount = items.filter(i => i.has_warning).length
  const avgJaccard = items.filter(i => i.factor_jaccard_sim !== null).length > 0
    ? items.reduce((sum, i) => sum + (i.factor_jaccard_sim || 0), 0) /
      items.filter(i => i.factor_jaccard_sim !== null).length
    : null
  const avgIcir = items.filter(i => i.icir_5w !== null).length > 0
    ? items.reduce((sum, i) => sum + (i.icir_5w || 0), 0) /
      items.filter(i => i.icir_5w !== null).length
    : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Training Quality</h1>
          <p className="text-muted-foreground">
            Monitor factor stability and IC consistency across training runs
          </p>
        </div>
        <button className="btn btn-secondary" onClick={() => fetchData()}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Training Runs</span>
            <div className="icon-box icon-box-blue">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{items.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Warnings</span>
            <div className={cn("icon-box", warningCount > 0 ? "icon-box-red" : "icon-box-green")}>
              {warningCount > 0 ? <AlertTriangle className="h-4 w-4" /> : <CheckCircle className="h-4 w-4" />}
            </div>
          </div>
          <p className={cn("stat-value", warningCount > 0 && "text-red")}>
            {warningCount}
          </p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Avg Jaccard</span>
            <div className="icon-box icon-box-purple">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className={cn(
            "stat-value",
            avgJaccard !== null && avgJaccard >= (thresholds?.jaccard_min || 0.3) ? "text-green" : "text-yellow-600"
          )}>
            {avgJaccard?.toFixed(3) || '---'}
          </p>
          <p className="text-[10px] text-muted-foreground">
            min: {thresholds?.jaccard_min || 0.3}
          </p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Avg ICIR (5w)</span>
            <div className="icon-box icon-box-orange">
              <TrendingUp className="h-4 w-4" />
            </div>
          </div>
          <p className={cn(
            "stat-value",
            avgIcir !== null && avgIcir >= (thresholds?.icir_min || 0.5) ? "text-green" : "text-yellow-600"
          )}>
            {avgIcir?.toFixed(3) || '---'}
          </p>
          <p className="text-[10px] text-muted-foreground">
            min: {thresholds?.icir_min || 0.5}
          </p>
        </div>
      </div>

      {/* Thresholds Info */}
      {thresholds && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue" />
              Quality Thresholds
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="p-3 rounded bg-secondary/50">
                <p className="text-muted-foreground text-xs">Jaccard Min</p>
                <p className="font-semibold">{thresholds.jaccard_min}</p>
                <p className="text-[10px] text-muted-foreground">Factor overlap between consecutive weeks</p>
              </div>
              <div className="p-3 rounded bg-secondary/50">
                <p className="text-muted-foreground text-xs">IC Std Max</p>
                <p className="font-semibold">{thresholds.ic_std_max}</p>
                <p className="text-[10px] text-muted-foreground">5-week moving IC standard deviation</p>
              </div>
              <div className="p-3 rounded bg-secondary/50">
                <p className="text-muted-foreground text-xs">ICIR Min</p>
                <p className="font-semibold">{thresholds.icir_min}</p>
                <p className="text-[10px] text-muted-foreground">5-week IC Information Ratio</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quality Table */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4 text-green" />
            Quality Metrics History
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Activity className="h-8 w-8 mb-2 opacity-50" />
              <p className="text-sm">No quality metrics recorded yet</p>
              <p className="text-xs">Metrics will appear after training runs complete</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-secondary/30">
                    <th className="text-left p-3 font-medium">Week</th>
                    <th className="text-right p-3 font-medium">Jaccard</th>
                    <th className="text-right p-3 font-medium">Overlap</th>
                    <th className="text-right p-3 font-medium">IC Avg (5w)</th>
                    <th className="text-right p-3 font-medium">IC Std (5w)</th>
                    <th className="text-right p-3 font-medium">ICIR (5w)</th>
                    <th className="text-center p-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <QualityRow key={item.training_run_id} item={item} thresholds={thresholds} />
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

function QualityRow({ item, thresholds }: { item: QualityMetricsItem; thresholds?: QualityResponse['thresholds'] }) {
  const jaccardOk = item.factor_jaccard_sim === null || item.factor_jaccard_sim >= (thresholds?.jaccard_min || 0.3)
  const icStdOk = item.ic_moving_std_5w === null || item.ic_moving_std_5w <= (thresholds?.ic_std_max || 0.1)
  const icirOk = item.icir_5w === null || item.icir_5w >= (thresholds?.icir_min || 0.5)

  return (
    <tr className={cn(
      "border-b hover:bg-secondary/20 transition-colors",
      item.has_warning && "bg-red/5"
    )}>
      <td className="p-3">
        <span className="font-mono font-medium">{item.week_id || `Run #${item.training_run_id}`}</span>
      </td>
      <td className={cn("p-3 text-right font-mono", !jaccardOk && "text-yellow-600")}>
        {item.factor_jaccard_sim?.toFixed(3) || '---'}
      </td>
      <td className="p-3 text-right font-mono">
        {item.factor_overlap_count ?? '---'}
      </td>
      <td className="p-3 text-right font-mono">
        {item.ic_moving_avg_5w?.toFixed(4) || '---'}
      </td>
      <td className={cn("p-3 text-right font-mono", !icStdOk && "text-yellow-600")}>
        {item.ic_moving_std_5w?.toFixed(4) || '---'}
      </td>
      <td className={cn("p-3 text-right font-mono", !icirOk && "text-yellow-600")}>
        {item.icir_5w?.toFixed(3) || '---'}
      </td>
      <td className="p-3 text-center">
        {item.has_warning ? (
          <div className="flex items-center justify-center gap-1 text-yellow-600" title={item.warning_message || ''}>
            <AlertTriangle className="h-4 w-4" />
            <span className="text-xs">{item.warning_type?.replace('_', ' ')}</span>
          </div>
        ) : (
          <CheckCircle className="h-4 w-4 text-green mx-auto" />
        )}
      </td>
    </tr>
  )
}
