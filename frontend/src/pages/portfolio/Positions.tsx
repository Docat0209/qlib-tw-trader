import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Briefcase, TrendingUp, TrendingDown, DollarSign, Package, Loader2, RefreshCw, Target } from 'lucide-react'
import { portfolioApi, PositionsResponse, PredictionsLatestResponse } from '@/api/client'

export function Positions() {
  const [positions, setPositions] = useState<PositionsResponse | null>(null)
  const [predictions, setPredictions] = useState<PredictionsLatestResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [posRes, predRes] = await Promise.all([
        portfolioApi.positions(),
        portfolioApi.predictionsLatest(),
      ])
      setPositions(posRes)
      setPredictions(predRes)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const formatCurrency = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`
    }
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(1)}K`
    }
    return `$${value.toFixed(0)}`
  }

  const formatPercent = (value: number | null) => {
    if (value === null) return '---'
    return `${(value * 100).toFixed(2)}%`
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

  const positionCount = positions?.positions.length || 0
  const totalUnrealizedPnl = positions?.positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0) || 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Positions</h1>
          <p className="subheading mt-1">View and manage your current holdings.</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary">
          <span className="dot dot-green" />
          <span className="text-sm font-medium">As of {positions?.as_of}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Market Value</span>
            <div className="icon-box icon-box-blue">
              <DollarSign className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{formatCurrency(positions?.total_value || 0)}</p>
          <p className="text-sm text-muted-foreground mt-2">NAV</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Positions</span>
            <div className="icon-box icon-box-green">
              <Package className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{positionCount}</p>
          <p className="text-sm text-muted-foreground mt-2">Active holdings</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Unrealized P/L</span>
            <div className={`icon-box ${totalUnrealizedPnl >= 0 ? 'icon-box-green' : 'icon-box-orange'}`}>
              {totalUnrealizedPnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            </div>
          </div>
          <p className={`stat-value ${totalUnrealizedPnl >= 0 ? 'text-green' : 'text-red'}`}>
            {totalUnrealizedPnl >= 0 ? '+' : ''}{formatCurrency(totalUnrealizedPnl)}
          </p>
          <p className="text-sm text-muted-foreground mt-2">Total unrealized</p>
        </div>
      </div>

      {/* Positions Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Briefcase className="h-4 w-4 text-blue" />
            Position Details
          </CardTitle>
          <span className="badge badge-blue">{positionCount} positions</span>
        </CardHeader>
        <CardContent className="p-0">
          {positionCount === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <Briefcase className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Positions</p>
              <p className="text-sm text-muted-foreground mt-1">Execute trades to see positions here</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Symbol</th>
                    <th className="table-header px-5 py-3 text-right">Shares</th>
                    <th className="table-header px-5 py-3 text-right">Cost</th>
                    <th className="table-header px-5 py-3 text-right">Price</th>
                    <th className="table-header px-5 py-3 text-right">Market Value</th>
                    <th className="table-header px-5 py-3 text-right">P/L</th>
                    <th className="table-header px-5 py-3 text-right">Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {positions?.positions.map((pos) => (
                    <tr key={pos.symbol} className="table-row">
                      <td className="table-cell px-5">
                        <div>
                          <span className="font-semibold">{pos.symbol}</span>
                          {pos.name && (
                            <p className="text-xs text-muted-foreground">{pos.name}</p>
                          )}
                        </div>
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {pos.shares.toLocaleString()}
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {pos.avg_cost.toFixed(2)}
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {pos.current_price?.toFixed(2) || '---'}
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {formatCurrency(pos.market_value || 0)}
                      </td>
                      <td className="table-cell px-5 text-right">
                        <div>
                          <span className={`mono font-semibold ${(pos.unrealized_pnl || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                            {(pos.unrealized_pnl || 0) >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl || 0)}
                          </span>
                          <p className={`text-xs ${(pos.unrealized_pnl_pct || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                            {formatPercent(pos.unrealized_pnl_pct)}
                          </p>
                        </div>
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {formatPercent(pos.weight)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Predictions */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Target className="h-4 w-4 text-purple" />
            Latest Predictions
          </CardTitle>
          <span className="badge badge-purple">{predictions?.date || '---'}</span>
        </CardHeader>
        <CardContent className="p-0">
          {!predictions || predictions.signals.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-purple w-12 h-12 mb-4">
                <Target className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Predictions</p>
              <p className="text-sm text-muted-foreground mt-1">Run model to generate predictions</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Rank</th>
                    <th className="table-header px-5 py-3 text-left">Symbol</th>
                    <th className="table-header px-5 py-3 text-right">Score</th>
                    <th className="table-header px-5 py-3 text-center">Signal</th>
                    <th className="table-header px-5 py-3 text-right">Current Position</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.signals.map((sig) => (
                    <tr key={sig.symbol} className="table-row">
                      <td className="table-cell px-5">
                        <span className="font-semibold">#{sig.rank}</span>
                      </td>
                      <td className="table-cell px-5">
                        <div>
                          <span className="font-semibold">{sig.symbol}</span>
                          {sig.name && (
                            <p className="text-xs text-muted-foreground">{sig.name}</p>
                          )}
                        </div>
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {sig.score.toFixed(4)}
                      </td>
                      <td className="table-cell px-5 text-center">
                        <span className={`badge ${
                          sig.signal === 'buy' ? 'badge-green' :
                          sig.signal === 'sell' ? 'badge-red' :
                          'badge-gray'
                        }`}>
                          {sig.signal}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right mono">
                        {sig.current_position?.toLocaleString() || 0}
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
