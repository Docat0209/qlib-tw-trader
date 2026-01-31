import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { TrendingUp, TrendingDown, Database, Brain, DollarSign, Activity, Loader2, RefreshCw, AlertTriangle } from 'lucide-react'
import { dashboardApi, DashboardSummary } from '@/api/client'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'

export function Dashboard() {
  const [data, setData] = useState<DashboardSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await dashboardApi.summary()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // 自動刷新（監聽 data_updated 事件）
  useFetchOnChange('dashboard', fetchData)

  const formatPercent = (value: number | null) => {
    if (value === null) return '---'
    const prefix = value >= 0 ? '+' : ''
    return `${prefix}${(value * 100).toFixed(2)}%`
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

  const today = new Date().toLocaleDateString('zh-TW', { year: 'numeric', month: '2-digit', day: '2-digit' })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Dashboard</h1>
          <p className="subheading mt-1">Overview of your trading system.</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`dot ${data?.data_status.is_complete ? 'dot-green' : 'dot-orange'}`} />
          <span className="text-sm text-muted-foreground">{data?.data_status.is_complete ? 'Data Complete' : 'Data Incomplete'}</span>
          <span className="text-sm text-muted-foreground mx-2">|</span>
          <span className="text-sm mono">{today}</span>
        </div>
      </div>

      {/* Model Retrain Warning */}
      {data?.model.needs_retrain && (
        <div className="flex items-center gap-3 p-4 rounded-lg bg-orange/10 border border-orange/20">
          <AlertTriangle className="h-5 w-5 text-orange" />
          <div>
            <p className="font-semibold text-orange">Model Retrain Recommended</p>
            <p className="text-sm text-muted-foreground">
              Last trained {data.model.days_since_training} days ago.
            </p>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Return</span>
            <div className={`icon-box ${(data?.performance.total_return || 0) >= 0 ? 'icon-box-green' : 'icon-box-orange'}`}>
              {(data?.performance.total_return || 0) >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            </div>
          </div>
          <p className={`stat-value ${(data?.performance.total_return || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatPercent(data?.performance.total_return || null)}
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            MTD: {formatPercent(data?.performance.mtd_return || null)}
          </p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Model IC</span>
            <div className="icon-box icon-box-blue">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value mono">{data?.model.ic?.toFixed(4) || '---'}</p>
          <p className="text-sm text-muted-foreground mt-2">
            ICIR: {data?.model.icir?.toFixed(2) || '---'}
          </p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Predictions</span>
            <div className="icon-box icon-box-purple">
              <DollarSign className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">
            <span className="text-green">{data?.prediction.buy_signals || 0}</span>
            <span className="text-muted-foreground mx-1">/</span>
            <span className="text-red">{data?.prediction.sell_signals || 0}</span>
          </p>
          <p className="text-sm text-muted-foreground mt-2">Buy / Sell signals</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Factors</span>
            <div className="icon-box icon-box-orange">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{data?.factors.enabled || 0}</p>
          <p className="text-sm text-muted-foreground mt-2">
            {data?.factors.low_selection_count || 0} low selection
          </p>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatusCard
          label="Data Status"
          value={data?.data_status.is_complete ? 'Complete' : `${data?.data_status.missing_count} Missing`}
          status={data?.data_status.is_complete ? 'green' : 'orange'}
        />
        <StatusCard
          label="Model"
          value={data?.model.factor_count ? `${data.model.factor_count} factors` : 'No model'}
          status={data?.model.factor_count ? 'blue' : 'gray'}
        />
        <StatusCard
          label="Top Pick"
          value={data?.prediction.top_pick?.symbol || '---'}
          status={data?.prediction.top_pick ? 'green' : 'gray'}
        />
        <StatusCard
          label="Days Since Train"
          value={data?.model.days_since_training?.toString() || '---'}
          status={data?.model.needs_retrain ? 'orange' : 'green'}
        />
      </div>

      {/* Content Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Model Info */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Current Model</CardTitle>
            {data?.model.ic && <span className="badge badge-green">Active</span>}
          </CardHeader>
          <CardContent className="space-y-3">
            {data?.model.ic ? (
              <>
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">IC</span>
                  <span className="font-semibold mono text-green">{data.model.ic.toFixed(4)}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">ICIR</span>
                  <span className="font-semibold mono">{data.model.icir?.toFixed(2) || '---'}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-muted-foreground">Factor Count</span>
                  <span className="font-semibold">{data.model.factor_count || '---'}</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm text-muted-foreground">Last Trained</span>
                  <span className="font-semibold mono text-sm">{data.model.last_trained_at?.split('T')[0] || '---'}</span>
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <Brain className="h-12 w-12 mb-4 opacity-50" />
                <p>No model trained yet</p>
                <p className="text-sm">Train a model to see metrics here.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Factor Overview */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Factor Overview</CardTitle>
            <span className="badge badge-blue">{data?.factors.total || 0} total</span>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">Total Factors</span>
              <span className="font-semibold">{data?.factors.total || 0}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">Enabled</span>
              <span className="font-semibold text-green">{data?.factors.enabled || 0}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">Disabled</span>
              <span className="font-semibold text-muted-foreground">{(data?.factors.total || 0) - (data?.factors.enabled || 0)}</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-muted-foreground">Low Selection Rate</span>
              <span className={`font-semibold ${(data?.factors.low_selection_count || 0) > 0 ? 'text-orange' : ''}`}>
                {data?.factors.low_selection_count || 0}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-3">
            <ActionButton label="Sync Data" icon={<Database className="h-5 w-5" />} color="blue" />
            <ActionButton label="Train Model" icon={<Brain className="h-5 w-5" />} color="purple" />
            <ActionButton label="View Factors" icon={<Activity className="h-5 w-5" />} color="green" />
            <ActionButton label="Predictions" icon={<TrendingUp className="h-5 w-5" />} color="orange" />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function StatusCard({ label, value, status }: { label: string; value: string; status: string }) {
  return (
    <div className="card p-4 flex items-center justify-between">
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="font-semibold">{value}</p>
      </div>
      <span className={`dot dot-${status}`} />
    </div>
  )
}

function ActionButton({ label, icon, color }: { label: string; icon: React.ReactNode; color: string }) {
  return (
    <button className={`btn btn-secondary flex-col gap-2 py-4 card-hover`}>
      <span className={`text-${color}`}>{icon}</span>
      <span className="text-xs">{label}</span>
    </button>
  )
}
