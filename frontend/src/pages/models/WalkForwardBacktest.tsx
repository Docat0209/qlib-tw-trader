import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Play,
  Clock,
  BarChart3,
  Loader2,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Trash2,
  Activity,
  AlertTriangle,
} from 'lucide-react'
import {
  walkForwardApi,
  WalkForwardItem,
  WalkForwardDetail,
  WeekStatus,
} from '@/api/client'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { EquityCurve } from '@/components/charts/EquityCurve'
import { WeekRangeSelector } from '@/components/WeekRangeSelector'

export function WalkForwardBacktest() {
  const [backtests, setBacktests] = useState<WalkForwardItem[]>([])
  const [selectedBacktest, setSelectedBacktest] = useState<WalkForwardDetail | null>(null)
  const [availableWeeks, setAvailableWeeks] = useState<WeekStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  // Form state
  const [startWeek, setStartWeek] = useState<string | null>(null)
  const [endWeek, setEndWeek] = useState<string | null>(null)
  const [initialCapital, setInitialCapital] = useState('1000000')
  const [maxPositions, setMaxPositions] = useState('10')
  const [tradePrice, setTradePrice] = useState<'close' | 'open'>('open')
  const [enableIncremental, setEnableIncremental] = useState(false)

  const { activeJob, cancelJob } = useJobs()
  const [actionLoading, setActionLoading] = useState(false)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [backtestsRes, weeksRes] = await Promise.all([
        walkForwardApi.list(50),
        walkForwardApi.availableWeeks(),
      ])
      setBacktests(backtestsRes.items)
      setAvailableWeeks(weeksRes.weeks)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleRun = async () => {
    if (!startWeek || !endWeek) {
      setError('Please select start and end weeks')
      return
    }

    setRunning(true)
    setError(null)
    try {
      await walkForwardApi.run({
        start_week_id: startWeek,
        end_week_id: endWeek,
        initial_capital: Number(initialCapital),
        max_positions: Number(maxPositions),
        trade_price: tradePrice,
        enable_incremental: enableIncremental,
        strategy: 'topk',
      })
      setTimeout(fetchData, 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start backtest')
    } finally {
      setRunning(false)
    }
  }

  const handleSelectBacktest = async (id: number) => {
    try {
      const detail = await walkForwardApi.get(id)
      setSelectedBacktest(detail)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load backtest detail')
    }
  }

  const handleDeleteBacktest = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation()
    if (!confirm('Delete this walk-forward backtest?')) return

    try {
      await walkForwardApi.delete(id)
      if (selectedBacktest?.id === id) {
        setSelectedBacktest(null)
      }
      fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete backtest')
    }
  }

  const handleRangeChange = (start: string | null, end: string | null) => {
    setStartWeek(start)
    setEndWeek(end)
  }

  const handleCancelBacktest = async () => {
    if (!activeJob) return
    setActionLoading(true)
    try {
      await cancelJob(activeJob.id)
      await fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel backtest')
    } finally {
      setActionLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useFetchOnChange('walk_forward_backtests', fetchData)

  useEffect(() => {
    if (activeJob?.job_type === 'walk_forward') {
      if (activeJob.status === 'completed') {
        fetchData()
      } else if (activeJob.status === 'failed') {
        setError(activeJob.error || 'Backtest failed')
        fetchData()
      }
    }
  }, [activeJob, fetchData])

  const formatPercent = (value: number | null | undefined) => {
    if (value === null || value === undefined) return '---'
    const prefix = value >= 0 ? '+' : ''
    return `${prefix}${value.toFixed(2)}%`
  }

  const formatIc = (value: number | null | undefined) => {
    if (value === null || value === undefined) return '---'
    return value.toFixed(4)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-100px)]">
      {/* Left Panel: Week Selector + History */}
      <div className="w-80 shrink-0 flex flex-col gap-4">
        {/* Week Range Selector */}
        <Card className="flex-1 overflow-hidden flex flex-col">
          <CardHeader className="shrink-0 py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue" />
              Select Weeks
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto pb-4">
            <WeekRangeSelector
              weeks={availableWeeks}
              startWeek={startWeek}
              endWeek={endWeek}
              onRangeChange={handleRangeChange}
            />
          </CardContent>
        </Card>

        {/* History */}
        <Card className="h-64 shrink-0 flex flex-col overflow-hidden">
          <CardHeader className="shrink-0 flex flex-row items-center justify-between py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Clock className="h-4 w-4 text-blue" />
              History
            </CardTitle>
            <div className="flex items-center gap-2">
              <span className="badge badge-gray text-xs">{backtests.length}</span>
              <button onClick={fetchData} className="p-1 hover:bg-secondary rounded">
                <RefreshCw className="h-3 w-3" />
              </button>
            </div>
          </CardHeader>
          <CardContent className="flex-1 p-0 overflow-y-auto">
            {backtests.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <Activity className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm">No backtests yet</p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {backtests.map((bt) => (
                  <div
                    key={bt.id}
                    onClick={() => handleSelectBacktest(bt.id)}
                    className={`w-full text-left p-3 hover:bg-secondary/50 transition-colors cursor-pointer group ${
                      selectedBacktest?.id === bt.id ? 'bg-blue-50 border-l-2 border-blue' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-sm">#{bt.id}</span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => handleDeleteBacktest(e, bt.id)}
                          className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red/10 transition-opacity"
                          title="Delete"
                        >
                          <Trash2 className="h-3 w-3 text-red" />
                        </button>
                        {bt.status === 'completed' ? (
                          <CheckCircle className="h-3 w-3 text-green" />
                        ) : bt.status === 'running' ? (
                          <Loader2 className="h-3 w-3 animate-spin text-blue" />
                        ) : bt.status === 'failed' ? (
                          <AlertCircle className="h-3 w-3 text-red" />
                        ) : (
                          <Clock className="h-3 w-3 text-gray-400" />
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {bt.start_week_id} ~ {bt.end_week_id}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Right Panel: Main Content */}
      <div className="flex-1 flex flex-col gap-4 overflow-y-auto pr-1">
        {/* Error */}
        {error && (
          <div className="p-3 rounded-lg bg-red-50 border border-red-100 shrink-0">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red" />
              <p className="text-sm text-red">{error}</p>
            </div>
          </div>
        )}

        {/* Active Job Progress */}
        {activeJob && activeJob.job_type === 'walk_forward' && ['queued', 'running'].includes(activeJob.status) && (
          <div className="p-4 rounded-lg bg-blue/10 border border-blue/20 shrink-0">
            <div className="flex items-center gap-3 mb-2">
              <Loader2 className="h-5 w-5 animate-spin text-blue" />
              <div className="flex-1">
                <p className="font-semibold text-blue">Walk-Forward Backtest in Progress</p>
                <p className="text-sm text-muted-foreground">
                  {activeJob.message || 'Processing...'}
                </p>
              </div>
              <span className="font-mono text-lg text-blue">
                {typeof activeJob.progress === 'number' ? activeJob.progress.toFixed(1) : activeJob.progress}%
              </span>
              <button
                className="btn btn-sm btn-ghost text-red hover:bg-red/10"
                onClick={handleCancelBacktest}
                disabled={actionLoading}
                title="Cancel backtest"
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

        {/* Row 1: Config + IC Analysis */}
        <div className="grid grid-cols-2 gap-4 shrink-0">
          {/* Configuration */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Play className="h-4 w-4 text-green" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 pt-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Capital</label>
                  <input
                    type="number"
                    className="input w-full text-sm"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Positions</label>
                  <input
                    type="number"
                    className="input w-full text-sm"
                    value={maxPositions}
                    onChange={(e) => setMaxPositions(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Trade Price</label>
                  <select
                    className="input w-full text-sm"
                    value={tradePrice}
                    onChange={(e) => setTradePrice(e.target.value as 'close' | 'open')}
                  >
                    <option value="open">Open</option>
                    <option value="close">Close</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Incremental</label>
                  <select
                    className="input w-full text-sm"
                    value={enableIncremental ? 'yes' : 'no'}
                    onChange={(e) => setEnableIncremental(e.target.value === 'yes')}
                  >
                    <option value="no">Disabled</option>
                    <option value="yes">Enabled</option>
                  </select>
                </div>
              </div>
              <button
                onClick={handleRun}
                disabled={running || !startWeek || !endWeek}
                className="btn btn-primary w-full text-sm disabled:opacity-50"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {running ? 'Starting...' : 'Run Walk-Forward Backtest'}
              </button>
            </CardContent>
          </Card>

          {/* IC Analysis */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4 text-blue" />
                IC Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {selectedBacktest?.ic_analysis ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <IcMetricCard
                      label="Avg Valid IC"
                      value={formatIc(selectedBacktest.ic_analysis.avg_valid_ic)}
                      sublabel="Training period"
                    />
                    <IcMetricCard
                      label="Avg Live IC"
                      value={formatIc(selectedBacktest.ic_analysis.avg_live_ic)}
                      sublabel="Prediction period"
                      highlight
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <IcMetricCard
                      label="IC Decay"
                      value={`${selectedBacktest.ic_analysis.ic_decay.toFixed(1)}%`}
                      sublabel="Degradation"
                      color={selectedBacktest.ic_analysis.ic_decay > 50 ? 'red' : selectedBacktest.ic_analysis.ic_decay > 30 ? 'yellow' : 'green'}
                    />
                    <IcMetricCard
                      label="IC Correlation"
                      value={selectedBacktest.ic_analysis.ic_correlation?.toFixed(2) || '---'}
                      sublabel="Valid vs Live"
                    />
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
                  Select a backtest to view IC analysis
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Row 2: Return Metrics */}
        <Card className="shrink-0">
          <CardHeader className="py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-blue" />
              Return Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            {selectedBacktest?.return_metrics ? (
              <div className="grid grid-cols-7 gap-3">
                <MetricCard
                  label="Cumulative"
                  value={formatPercent(selectedBacktest.return_metrics.cumulative_return)}
                  color={(selectedBacktest.return_metrics.cumulative_return || 0) >= 0}
                />
                <MetricCard
                  label="Market"
                  value={formatPercent(selectedBacktest.return_metrics.market_return)}
                />
                <MetricCard
                  label="Excess"
                  value={formatPercent(selectedBacktest.return_metrics.excess_return)}
                  color={(selectedBacktest.return_metrics.excess_return || 0) >= 0}
                  highlight
                />
                <MetricCard
                  label="Sharpe"
                  value={selectedBacktest.return_metrics.sharpe_ratio?.toFixed(2) || '---'}
                />
                <MetricCard
                  label="Max DD"
                  value={selectedBacktest.return_metrics.max_drawdown ? `-${selectedBacktest.return_metrics.max_drawdown.toFixed(1)}%` : '---'}
                  color={false}
                />
                <MetricCard
                  label="Win Rate"
                  value={selectedBacktest.return_metrics.win_rate ? `${selectedBacktest.return_metrics.win_rate.toFixed(0)}%` : '---'}
                />
                <MetricCard
                  label="Trades"
                  value={selectedBacktest.return_metrics.total_trades?.toString() || '---'}
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-16 text-muted-foreground text-sm">
                Select a backtest to view return metrics
              </div>
            )}
          </CardContent>
        </Card>

        {/* Row 3: Equity Curve */}
        <Card className="shrink-0">
          <CardHeader className="py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-blue" />
              Equity Curve
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            {selectedBacktest?.equity_curve && selectedBacktest.equity_curve.length > 0 ? (
              <EquityCurve
                data={selectedBacktest.equity_curve}
                initialCapital={selectedBacktest.config.initial_capital}
                height={160}
              />
            ) : (
              <div className="flex items-center justify-center h-[160px] text-muted-foreground text-sm">
                <TrendingUp className="h-8 w-8 mr-3 opacity-30" />
                Select a backtest to view equity curve
              </div>
            )}
          </CardContent>
        </Card>

        {/* Row 4: Weekly Details */}
        <Card className="flex-1 min-h-[300px] flex flex-col overflow-hidden">
          <CardHeader className="py-3 shrink-0">
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue" />
              Weekly Details
              {selectedBacktest?.weekly_details && (
                <span className="text-xs text-muted-foreground font-normal">
                  ({selectedBacktest.weekly_details.length} weeks)
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 pt-0 overflow-hidden">
            {selectedBacktest?.weekly_details && selectedBacktest.weekly_details.length > 0 ? (
              <div className="h-full overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-background">
                    <tr className="border-b text-left">
                      <th className="py-2 px-2 font-medium">Predict Week</th>
                      <th className="py-2 px-2 font-medium">Model</th>
                      <th className="py-2 px-2 font-medium text-right">Valid IC</th>
                      <th className="py-2 px-2 font-medium text-right">Live IC</th>
                      <th className="py-2 px-2 font-medium text-right">Decay</th>
                      <th className="py-2 px-2 font-medium text-right">Incr</th>
                      <th className="py-2 px-2 font-medium text-right">Return</th>
                      <th className="py-2 px-2 font-medium text-right">Market</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedBacktest.weekly_details.map((week) => (
                      <tr
                        key={week.predict_week}
                        className={`border-b hover:bg-secondary/50 ${week.is_fallback ? 'bg-yellow-50' : ''}`}
                      >
                        <td className="py-2 px-2 font-mono">{week.predict_week}</td>
                        <td className="py-2 px-2">
                          <span className="font-mono text-xs">{week.model_week}</span>
                          {week.is_fallback && (
                            <span className="ml-1 text-[10px] text-yellow-600 bg-yellow-100 px-1 rounded">
                              fallback
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-2 text-right font-mono">
                          {formatIc(week.valid_ic)}
                        </td>
                        <td className="py-2 px-2 text-right font-mono">
                          {formatIc(week.live_ic)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${
                          (week.ic_decay || 0) > 50 ? 'text-red' : (week.ic_decay || 0) > 30 ? 'text-yellow-600' : 'text-green'
                        }`}>
                          {week.ic_decay !== null ? `${week.ic_decay.toFixed(0)}%` : '---'}
                        </td>
                        <td className="py-2 px-2 text-right font-mono text-muted-foreground">
                          {week.incremental_days !== null ? `${week.incremental_days}d` : '---'}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${
                          (week.week_return || 0) >= 0 ? 'text-green' : 'text-red'
                        }`}>
                          {formatPercent(week.week_return)}
                        </td>
                        <td className="py-2 px-2 text-right font-mono text-muted-foreground">
                          {formatPercent(week.market_return)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
                <BarChart3 className="h-12 w-12 mb-3 opacity-30" />
                Select a backtest to view weekly details
              </div>
            )}
          </CardContent>
        </Card>
      </div>

    </div>
  )
}

// IC Metric Card
function IcMetricCard({
  label,
  value,
  sublabel,
  highlight,
  color,
}: {
  label: string
  value: string
  sublabel?: string
  highlight?: boolean
  color?: 'red' | 'yellow' | 'green'
}) {
  const colorClass = color === 'red' ? 'text-red' : color === 'yellow' ? 'text-yellow-600' : color === 'green' ? 'text-green' : ''

  return (
    <div className={`p-3 rounded ${highlight ? 'bg-blue/10 border border-blue/20' : 'bg-secondary/50'}`}>
      <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold font-mono ${colorClass}`}>{value}</p>
      {sublabel && <p className="text-[10px] text-muted-foreground">{sublabel}</p>}
    </div>
  )
}

// Metric Card
function MetricCard({
  label,
  value,
  color,
  highlight,
}: {
  label: string
  value: string
  color?: boolean
  highlight?: boolean
}) {
  return (
    <div className={`p-2 rounded text-center ${highlight ? 'bg-blue/10 border border-blue/20' : 'bg-secondary/50'}`}>
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={`font-semibold text-sm ${
        color === true ? 'text-green' : color === false ? 'text-red' : ''
      }`}>
        {value}
      </p>
    </div>
  )
}
