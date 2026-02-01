import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Play, Clock, BarChart3, Loader2, RefreshCw, AlertCircle, CheckCircle, TrendingUp, List } from 'lucide-react'
import {
  backtestApi,
  BacktestItem,
  BacktestDetail,
  modelApi,
  ModelHistoryItem,
  StockTradeInfo,
  StockKlineResponse,
  TradePoint,
  AllTradesResponse,
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

  // 全局交易記錄
  const [allTrades, setAllTrades] = useState<AllTradesResponse | null>(null)

  // Form state
  const [modelId, setModelId] = useState<string>('')
  const [initialCapital, setInitialCapital] = useState('1000000')
  const [maxPositions, setMaxPositions] = useState('10')
  const [tradePrice, setTradePrice] = useState<'close' | 'open'>('close')

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
        backtestApi.list(undefined, 50),
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
        trade_price: tradePrice,
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
      setSelectedStock(null)
      setStockKline(null)
      setAllTrades(null)

      // 載入交易過的股票和所有交易記錄
      if (detail.status === 'completed') {
        try {
          const [stocksRes, tradesRes] = await Promise.all([
            backtestApi.getStocks(id),
            backtestApi.getAllTrades(id),
          ])
          setTradedStocks(stocksRes.items)
          setAllTrades(tradesRes)

          // 自動選擇第一個股票
          if (stocksRes.items.length > 0) {
            const firstStock = stocksRes.items[0].stock_id
            setSelectedStock(firstStock)
            // 載入第一個股票的 K 線
            const klineRes = await backtestApi.getStockKline(id, firstStock)
            setStockKline(klineRes)
          }
        } catch {
          setTradedStocks([])
          setAllTrades(null)
        }
      } else {
        setTradedStocks([])
        setAllTrades(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load backtest detail')
    }
  }

  const handleSelectStock = async (stockId: string) => {
    if (!selectedBacktest || !stockId) {
      setStockKline(null)
      setSelectedStock(null)
      return
    }

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

  const formatPnl = (pnl: number | null | undefined) => {
    if (pnl === null || pnl === undefined) return null
    const prefix = pnl >= 0 ? '+' : ''
    return `${prefix}${pnl.toLocaleString()}`
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
      {/* 左側：Backtest History */}
      <div className="w-72 shrink-0 flex flex-col">
        <Card className="flex-1 flex flex-col overflow-hidden">
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
                <BarChart3 className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm">No backtests yet</p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {backtests.map((bt) => (
                  <button
                    key={bt.id}
                    onClick={() => handleSelectBacktest(bt.id)}
                    className={`w-full text-left p-3 hover:bg-secondary/50 transition-colors ${
                      selectedBacktest?.id === bt.id ? 'bg-blue-50 border-l-2 border-blue' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold">#{bt.id}</span>
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
                    <div className="text-xs text-muted-foreground mb-1">
                      {bt.start_date} ~ {bt.end_date}
                    </div>
                    {bt.metrics && (
                      <div className={`text-sm font-semibold ${
                        (bt.metrics.total_return_with_cost || 0) >= 0 ? 'text-green' : 'text-red'
                      }`}>
                        {formatPercent(bt.metrics.total_return_with_cost)}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 右側：主內容區 */}
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
        {activeJob && activeJob.job_type === 'backtest' && ['queued', 'running'].includes(activeJob.status) && (
          <Card className="shrink-0">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Loader2 className="h-4 w-4 animate-spin text-blue" />
                <div className="flex-1">
                  <p className="text-sm font-semibold">Running Backtest</p>
                  <p className="text-xs text-muted-foreground">{activeJob.message || 'Processing...'}</p>
                </div>
                <span className="text-lg font-semibold text-blue">{activeJob.progress}%</span>
              </div>
              <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue transition-all duration-300"
                  style={{ width: `${activeJob.progress}%` }}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Row 1: Run Form + Metrics */}
        <div className="grid grid-cols-2 gap-4 shrink-0">
          {/* Run New Backtest */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Play className="h-4 w-4 text-green" />
                Run Backtest
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 pt-0">
              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2">
                  <label className="text-xs text-muted-foreground block mb-1">Model</label>
                  <select
                    className="input w-full text-sm"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                  >
                    <option value="">Select model...</option>
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.id} - IC: {m.metrics.ic?.toFixed(4) || '---'}
                      </option>
                    ))}
                  </select>
                </div>
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
                <div className="col-span-2">
                  <label className="text-xs text-muted-foreground block mb-1">Trade Price</label>
                  <select
                    className="input w-full text-sm"
                    value={tradePrice}
                    onChange={(e) => setTradePrice(e.target.value as 'close' | 'open')}
                  >
                    <option value="close">Close Price (Default)</option>
                    <option value="open">Open Price (09:00)</option>
                  </select>
                </div>
              </div>
              <button
                onClick={handleRun}
                disabled={running || !modelId}
                className="btn btn-primary w-full text-sm disabled:opacity-50"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {running ? 'Starting...' : 'Run'}
              </button>
            </CardContent>
          </Card>

          {/* Metrics */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-blue" />
                {selectedBacktest ? `#${selectedBacktest.id} Metrics` : 'Metrics'}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              {selectedBacktest?.metrics ? (
                <div className="grid grid-cols-4 gap-2">
                  <MetricCard label="Return" value={formatPercent(selectedBacktest.metrics.total_return_with_cost)} color={(selectedBacktest.metrics.total_return_with_cost || 0) >= 0} />
                  <MetricCard label="No Cost" value={formatPercent(selectedBacktest.metrics.total_return_without_cost)} color={(selectedBacktest.metrics.total_return_without_cost || 0) >= 0} />
                  <MetricCard label="Sharpe" value={selectedBacktest.metrics.sharpe_ratio?.toFixed(2) || '---'} />
                  <MetricCard label="Max DD" value={`-${selectedBacktest.metrics.max_drawdown?.toFixed(1) || '---'}%`} color={false} />
                  <MetricCard label="Win Rate" value={`${selectedBacktest.metrics.win_rate?.toFixed(0) || '---'}%`} />
                  <MetricCard label="Trades" value={String(selectedBacktest.metrics.total_trades || '---')} />
                  <MetricCard label="Cost" value={`$${selectedBacktest.metrics.total_cost?.toFixed(0) || '---'}`} />
                  <MetricCard label="Period" value={selectedBacktest.start_date?.slice(5) || '---'} small />
                </div>
              ) : (
                <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
                  Select a backtest to view metrics
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Equity Curve */}
        <Card className="shrink-0">
          <CardHeader className="py-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-blue" />
              Equity Curve
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            {selectedBacktest?.equity_curve && selectedBacktest.equity_curve.length > 0 ? (
              <EquityCurve
                data={selectedBacktest.equity_curve}
                initialCapital={selectedBacktest.initial_capital}
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

        {/* K-line + Trade Details */}
        <div className="grid grid-cols-3 gap-4 flex-1 min-h-[400px]">
          {/* K-line Chart */}
          <Card className="col-span-2 flex flex-col">
            <CardHeader className="py-3 shrink-0">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <BarChart3 className="h-4 w-4 text-blue" />
                  K-line
                </CardTitle>
                <select
                  className="input text-sm w-48"
                  value={selectedStock || ''}
                  onChange={(e) => handleSelectStock(e.target.value)}
                  disabled={tradedStocks.length === 0}
                >
                  {tradedStocks.length === 0 ? (
                    <option value="">No stocks</option>
                  ) : (
                    tradedStocks.map((s) => (
                      <option key={s.stock_id} value={s.stock_id}>
                        {s.stock_id} {s.name} (B:{s.buy_count} S:{s.sell_count})
                      </option>
                    ))
                  )}
                </select>
              </div>
            </CardHeader>
            <CardContent className="flex-1 pt-0">
              {loadingKline ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : stockKline && stockKline.klines.length > 0 ? (
                <StockKlineChart
                  klines={stockKline.klines}
                  trades={stockKline.trades}
                  height={340}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
                  <BarChart3 className="h-12 w-12 mb-3 opacity-30" />
                  {tradedStocks.length > 0 ? 'Select a stock to view K-line' : 'No traded stocks'}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Trade Details */}
          <Card className="flex flex-col overflow-hidden">
            <CardHeader className="py-3 shrink-0">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <List className="h-4 w-4 text-blue" />
                  Trades
                  {allTrades && (
                    <span className="text-xs text-muted-foreground font-normal">
                      ({allTrades.total})
                    </span>
                  )}
                </CardTitle>
                {allTrades && (
                  <span className={`text-sm font-semibold ${allTrades.total_pnl >= 0 ? 'text-green' : 'text-red'}`}>
                    {allTrades.total_pnl >= 0 ? '+' : ''}{allTrades.total_pnl.toLocaleString()}
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent className="flex-1 pt-0 px-2 overflow-hidden">
              {allTrades && allTrades.items.length > 0 ? (
                <div className="h-full overflow-y-auto pr-1">
                  <TradeList trades={allTrades.items} showStock />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
                  <List className="h-8 w-8 mb-2 opacity-30" />
                  Select a backtest to view trades
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

// 指標卡片組件
function MetricCard({ label, value, color, small }: { label: string; value: string; color?: boolean; small?: boolean }) {
  return (
    <div className="p-2 rounded bg-secondary/50">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={`font-semibold ${small ? 'text-xs' : 'text-sm'} ${
        color === true ? 'text-green' : color === false ? 'text-red' : ''
      }`}>
        {value}
      </p>
    </div>
  )
}

// 交易列表組件
function TradeList({ trades, showStock }: { trades: TradePoint[]; showStock?: boolean }) {
  return (
    <div className="space-y-2">
      {trades.map((t, i) => (
        <div
          key={i}
          className={`p-2 rounded-lg text-xs ${
            t.side === 'buy' ? 'bg-green/5 border-l-2 border-green' : 'bg-red/5 border-l-2 border-red'
          }`}
        >
          <div className="flex justify-between items-center">
            <span className="font-semibold">
              {t.date.slice(5)} {t.side === 'buy' ? '買' : '賣'}
              {showStock && t.stock_id && (
                <span className="ml-1 text-muted-foreground font-normal">
                  {t.stock_id}
                </span>
              )}
            </span>
            {t.pnl !== null && t.pnl !== undefined && (
              <span className={`font-mono font-semibold ${t.pnl >= 0 ? 'text-green' : 'text-red'}`}>
                {t.pnl >= 0 ? '+' : ''}{t.pnl.toLocaleString()}
                {t.pnl_pct !== null && t.pnl_pct !== undefined && (
                  <span className="ml-1">({t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%)</span>
                )}
              </span>
            )}
          </div>
          <div className="text-muted-foreground mt-0.5">
            ${t.price.toLocaleString()} × {t.shares?.toLocaleString() || 0} 股
            {t.amount && <span className="ml-1">= ${t.amount.toLocaleString()}</span>}
          </div>
          {t.holding_days !== null && t.holding_days !== undefined && (
            <div className="text-muted-foreground">
              持有 {t.holding_days} 天
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
