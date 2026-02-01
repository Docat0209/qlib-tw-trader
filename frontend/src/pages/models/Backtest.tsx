import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Play, Clock, BarChart3, Loader2, RefreshCw, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react'
import {
  backtestApi,
  BacktestItem,
  BacktestDetail,
  modelApi,
  ModelHistoryItem,
  StockTradeInfo,
  StockKlineResponse,
} from '@/api/client'
import { useJobs } from '@/hooks/useJobs'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'
import { EquityCurve } from '@/components/charts/EquityCurve'
import { StockKlineChart } from '@/components/charts/StockKlineChart'

export function Backtest() {
  const [backtests, setBacktests] = useState<BacktestItem[]>([])
  const [models, setModels] = useState<ModelHistoryItem[]>([])
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  // 股票交易相關
  const [tradedStocks, setTradedStocks] = useState<StockTradeInfo[]>([])
  const [selectedStock, setSelectedStock] = useState<string | null>(null)
  const [stockKline, setStockKline] = useState<StockKlineResponse | null>(null)
  const [loadingKline, setLoadingKline] = useState(false)

  // Form state
  const [modelId, setModelId] = useState<string>('')
  const [initialCapital, setInitialCapital] = useState('1000000')
  const [maxPositions, setMaxPositions] = useState('10')

  // 將 "m001" 格式轉為數字 ID
  const parseModelId = (id: string): number => {
    if (id.startsWith('m')) {
      return parseInt(id.slice(1), 10)
    }
    return parseInt(id, 10)
  }

  const { activeJob } = useJobs()

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [backtestsRes, modelsRes] = await Promise.all([
        backtestApi.list(undefined, 20),
        modelApi.history(20),
      ])
      setBacktests(backtestsRes.items)
      setModels(modelsRes.items)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleRun = async () => {
    if (!modelId) {
      setError('Please select a model')
      return
    }

    setRunning(true)
    setError(null)
    try {
      await backtestApi.run({
        model_id: parseModelId(modelId),
        initial_capital: Number(initialCapital),
        max_positions: Number(maxPositions),
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
      const detail = await backtestApi.get(id)
      setSelectedBacktest(detail)

      // 載入交易過的股票
      if (detail.status === 'completed') {
        try {
          const stocksRes = await backtestApi.getStocks(id)
          setTradedStocks(stocksRes.items)
          setSelectedStock(null)
          setStockKline(null)
        } catch {
          setTradedStocks([])
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load backtest detail')
    }
  }

  const handleSelectStock = async (stockId: string) => {
    if (!selectedBacktest) return

    setSelectedStock(stockId)
    setLoadingKline(true)
    try {
      const klineRes = await backtestApi.getStockKline(selectedBacktest.id, stockId)
      setStockKline(klineRes)
    } catch (err) {
      console.error('Failed to load K-line:', err)
      setStockKline(null)
    } finally {
      setLoadingKline(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useFetchOnChange('backtests', fetchData)

  useEffect(() => {
    if (activeJob?.job_type === 'backtest') {
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Backtest</h1>
          <p className="subheading mt-1">Run backtests to evaluate model performance.</p>
        </div>
        <button onClick={fetchData} className="btn btn-secondary">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-100">
          <div className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-red" />
            <p className="text-red">{error}</p>
          </div>
        </div>
      )}

      {/* Active Job Progress */}
      {activeJob && activeJob.job_type === 'backtest' && ['queued', 'running'].includes(activeJob.status) && (
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center gap-4">
              <Loader2 className="h-5 w-5 animate-spin text-blue" />
              <div className="flex-1">
                <p className="font-semibold">Running Backtest</p>
                <p className="text-sm text-muted-foreground">{activeJob.message || 'Processing...'}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-semibold text-blue">{activeJob.progress}%</p>
              </div>
            </div>
            <div className="mt-3 h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-blue transition-all duration-300"
                style={{ width: `${activeJob.progress}%` }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Row: Run + Equity Curve */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Run New Backtest */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play className="h-4 w-4 text-green" />
              Run New Backtest
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Model *</label>
              <select
                className="input w-full"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              >
                <option value="">Select a model...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} - IC: {m.metrics.ic?.toFixed(4) || '---'} ({m.trained_at.split('T')[0]})
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Backtest period: valid_end + 1 month
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Initial Capital</label>
                <input
                  type="number"
                  className="input w-full"
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Max Positions</label>
                <input
                  type="number"
                  className="input w-full"
                  value={maxPositions}
                  onChange={(e) => setMaxPositions(e.target.value)}
                />
              </div>
            </div>
            <button
              onClick={handleRun}
              disabled={running || !modelId}
              className="btn btn-primary w-full disabled:opacity-50"
            >
              {running ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {running ? 'Starting...' : 'Run Backtest'}
            </button>
          </CardContent>
        </Card>

        {/* Equity Curve Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-blue" />
              Equity Curve
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedBacktest?.equity_curve && selectedBacktest.equity_curve.length > 0 ? (
              <EquityCurve
                data={selectedBacktest.equity_curve}
                initialCapital={selectedBacktest.initial_capital}
                height={180}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-[180px] text-muted-foreground">
                <TrendingUp className="h-12 w-12 mb-4 opacity-50" />
                <p>Select a backtest to view equity curve</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Metrics Summary */}
      {selectedBacktest && selectedBacktest.metrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-blue" />
              Backtest #{selectedBacktest.id} Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Return (with cost)</p>
                <p className={`text-xl font-semibold ${(selectedBacktest.metrics.total_return_with_cost || selectedBacktest.metrics.total_return || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                  {formatPercent(selectedBacktest.metrics.total_return_with_cost || selectedBacktest.metrics.total_return)}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Return (no cost)</p>
                <p className={`text-xl font-semibold ${(selectedBacktest.metrics.total_return_without_cost || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                  {formatPercent(selectedBacktest.metrics.total_return_without_cost)}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
                <p className="text-xl font-semibold">{selectedBacktest.metrics.sharpe_ratio?.toFixed(2) || '---'}</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Max Drawdown</p>
                <p className="text-xl font-semibold text-red">-{selectedBacktest.metrics.max_drawdown?.toFixed(2) || '---'}%</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Win Rate</p>
                <p className="font-semibold">{selectedBacktest.metrics.win_rate?.toFixed(1) || '---'}%</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Total Trades</p>
                <p className="font-semibold">{selectedBacktest.metrics.total_trades || '---'}</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Total Cost</p>
                <p className="font-semibold">${selectedBacktest.metrics.total_cost?.toLocaleString() || '---'}</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Period</p>
                <p className="font-semibold text-sm">{selectedBacktest.start_date} ~ {selectedBacktest.end_date}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stock K-line Section */}
      {selectedBacktest && selectedBacktest.status === 'completed' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-blue" />
              Stock K-line + Trades
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              {/* Stock List */}
              <div className="w-48 shrink-0 border-r pr-4">
                <p className="text-xs text-muted-foreground mb-2">Traded Stocks ({tradedStocks.length})</p>
                <div className="space-y-1 max-h-[400px] overflow-y-auto">
                  {tradedStocks.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No trades</p>
                  ) : (
                    tradedStocks.map((stock) => (
                      <button
                        key={stock.stock_id}
                        onClick={() => handleSelectStock(stock.stock_id)}
                        className={`w-full text-left p-2 rounded text-sm hover:bg-secondary transition-colors ${
                          selectedStock === stock.stock_id ? 'bg-blue-50 text-blue' : ''
                        }`}
                      >
                        <span className="font-semibold">{stock.stock_id}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                          B:{stock.buy_count} S:{stock.sell_count}
                        </span>
                      </button>
                    ))
                  )}
                </div>
              </div>

              {/* K-line Chart */}
              <div className="flex-1">
                {loadingKline ? (
                  <div className="flex items-center justify-center h-[400px]">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : stockKline ? (
                  <div>
                    <p className="text-sm font-semibold mb-2">
                      {stockKline.stock_id} - {stockKline.name}
                    </p>
                    <StockKlineChart
                      klines={stockKline.klines}
                      trades={stockKline.trades}
                      height={380}
                    />
                  </div>
                ) : (
                  <StockKlineChart klines={[]} trades={[]} height={400} />
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Backtest History */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue" />
            Backtest History
          </CardTitle>
          <span className="badge badge-gray">{backtests.length} records</span>
        </CardHeader>
        <CardContent className="p-0">
          {backtests.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <BarChart3 className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Backtests Yet</p>
              <p className="text-sm text-muted-foreground mt-1">Run a backtest to see results here</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">ID</th>
                    <th className="table-header px-5 py-3 text-left">Period</th>
                    <th className="table-header px-5 py-3 text-right">Return (w/cost)</th>
                    <th className="table-header px-5 py-3 text-right">Return (no cost)</th>
                    <th className="table-header px-5 py-3 text-right">Sharpe</th>
                    <th className="table-header px-5 py-3 text-right">Max DD</th>
                    <th className="table-header px-5 py-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {backtests.map((bt) => (
                    <tr
                      key={bt.id}
                      className={`table-row cursor-pointer ${selectedBacktest?.id === bt.id ? 'bg-blue-50' : ''}`}
                      onClick={() => handleSelectBacktest(bt.id)}
                    >
                      <td className="table-cell px-5">
                        <span className="font-semibold">#{bt.id}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="text-sm">{bt.start_date} ~ {bt.end_date}</span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className={`font-semibold ${(bt.metrics?.total_return_with_cost || bt.metrics?.total_return || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                          {formatPercent(bt.metrics?.total_return_with_cost || bt.metrics?.total_return)}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className={`font-semibold ${(bt.metrics?.total_return_without_cost || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                          {formatPercent(bt.metrics?.total_return_without_cost)}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="mono">{bt.metrics?.sharpe_ratio?.toFixed(2) || '---'}</span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="text-red">-{bt.metrics?.max_drawdown?.toFixed(2) || '---'}%</span>
                      </td>
                      <td className="table-cell px-5 text-center">
                        {bt.status === 'completed' ? (
                          <span className="badge badge-green">
                            <CheckCircle className="h-3 w-3" />
                            Done
                          </span>
                        ) : bt.status === 'running' ? (
                          <span className="badge badge-blue">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Running
                          </span>
                        ) : bt.status === 'failed' ? (
                          <span className="badge badge-red">
                            <AlertCircle className="h-3 w-3" />
                            Failed
                          </span>
                        ) : (
                          <span className="badge badge-gray">Queued</span>
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
