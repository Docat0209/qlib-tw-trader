import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { TrendingUp, Percent, Activity, ArrowDownRight, BarChart3, Loader2, RefreshCw } from 'lucide-react'
import { performanceApi, PerformanceSummary } from '@/api/client'

export function Performance() {
  const [data, setData] = useState<PerformanceSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await performanceApi.summary()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Performance</h1>
          <p className="subheading mt-1">Analyze your portfolio returns and metrics.</p>
        </div>
        <div className="text-sm text-muted-foreground">
          As of: <span className="mono">{data?.as_of || '---'}</span>
        </div>
      </div>

      {/* Main Stats */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="stat-card">
          <div className="flex items-center justify-between mb-2">
            <span className="stat-label">Total Return</span>
            <div className={`icon-box ${(data?.returns.total || 0) >= 0 ? 'icon-box-green' : 'icon-box-red'}`}>
              <TrendingUp className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${(data?.returns.total || 0) >= 0 ? 'text-green' : 'text-red'}`}>
            {formatPercent(data?.returns.total || null)}
          </p>
          <p className="text-sm text-muted-foreground mt-2">Since inception</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between mb-2">
            <span className="stat-label">YTD Return</span>
            <div className={`icon-box ${(data?.returns.ytd || 0) >= 0 ? 'icon-box-blue' : 'icon-box-red'}`}>
              <Percent className="h-4 w-4" />
            </div>
          </div>
          <p className={`stat-value ${(data?.returns.ytd || 0) >= 0 ? 'text-blue' : 'text-red'}`}>
            {formatPercent(data?.returns.ytd || null)}
          </p>
          <p className="text-sm text-muted-foreground mt-2">Year to date</p>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="card p-4 flex items-center gap-3">
          <span className={`dot ${(data?.returns.today || 0) >= 0 ? 'dot-green' : 'dot-red'}`} />
          <div>
            <p className="text-xs text-muted-foreground">Today</p>
            <p className={`text-xl font-semibold ${(data?.returns.today || 0) >= 0 ? 'text-green' : 'text-red'}`}>
              {formatPercent(data?.returns.today || null)}
            </p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className={`dot ${(data?.returns.wtd || 0) >= 0 ? 'dot-blue' : 'dot-red'}`} />
          <div>
            <p className="text-xs text-muted-foreground">WTD</p>
            <p className={`text-xl font-semibold ${(data?.returns.wtd || 0) >= 0 ? 'text-blue' : 'text-red'}`}>
              {formatPercent(data?.returns.wtd || null)}
            </p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className={`dot ${(data?.returns.mtd || 0) >= 0 ? 'dot-purple' : 'dot-red'}`} />
          <div>
            <p className="text-xs text-muted-foreground">MTD</p>
            <p className={`text-xl font-semibold ${(data?.returns.mtd || 0) >= 0 ? 'text-purple' : 'text-red'}`}>
              {formatPercent(data?.returns.mtd || null)}
            </p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className={`dot ${(data?.alpha?.ytd || 0) >= 0 ? 'dot-green' : 'dot-red'}`} />
          <div>
            <p className="text-xs text-muted-foreground">Alpha (YTD)</p>
            <p className={`text-xl font-semibold ${(data?.alpha?.ytd || 0) >= 0 ? 'text-green' : 'text-red'}`}>
              {formatPercent(data?.alpha?.ytd || null)}
            </p>
          </div>
        </div>
      </div>

      {/* Equity Curve */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-blue" />
            Equity Curve
          </CardTitle>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2">
              <span className="dot dot-blue" />
              <span className="text-xs text-muted-foreground">Portfolio</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="dot dot-gray" />
              <span className="text-xs text-muted-foreground">Benchmark</span>
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center rounded-lg bg-secondary">
            <div className="text-center">
              <div className="icon-box icon-box-blue w-12 h-12 mx-auto mb-4">
                <BarChart3 className="h-6 w-6" />
              </div>
              <p className="font-semibold">Chart Coming Soon</p>
              <p className="text-sm text-muted-foreground mt-1">Equity curve visualization</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue" />
              Returns vs Benchmark
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <MetricRow label="Today" value={formatPercent(data?.returns.today || null)} benchmark={formatPercent(data?.benchmark_returns.today || null)} />
            <MetricRow label="WTD" value={formatPercent(data?.returns.wtd || null)} benchmark={formatPercent(data?.benchmark_returns.wtd || null)} />
            <MetricRow label="MTD" value={formatPercent(data?.returns.mtd || null)} benchmark={formatPercent(data?.benchmark_returns.mtd || null)} />
            <MetricRow label="YTD" value={formatPercent(data?.returns.ytd || null)} benchmark={formatPercent(data?.benchmark_returns.ytd || null)} highlight />
            <MetricRow label="Total" value={formatPercent(data?.returns.total || null)} benchmark={formatPercent(data?.benchmark_returns.total || null)} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ArrowDownRight className="h-4 w-4 text-blue" />
              Alpha
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <MetricRow label="MTD Alpha" value={formatPercent(data?.alpha?.mtd || null)} highlight={(data?.alpha?.mtd || 0) > 0} />
            <MetricRow label="YTD Alpha" value={formatPercent(data?.alpha?.ytd || null)} highlight={(data?.alpha?.ytd || 0) > 0} />
            <div className="mt-4 p-4 rounded-lg bg-secondary">
              <p className="text-sm text-muted-foreground">
                Alpha measures your portfolio's excess return compared to the benchmark (TAIEX).
                Positive alpha indicates outperformance.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function MetricRow({ label, value, benchmark, highlight }: { label: string; value: string; benchmark?: string; highlight?: boolean }) {
  return (
    <div className={`flex items-center justify-between px-3 py-2.5 rounded-lg ${
      highlight ? 'bg-blue-50' : 'hover:bg-secondary'
    }`}>
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center gap-4">
        {benchmark && (
          <span className="text-sm text-muted-foreground mono">{benchmark}</span>
        )}
        <span className={`font-semibold mono ${highlight ? 'text-blue' : ''}`}>{value}</span>
      </div>
    </div>
  )
}
