import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { TrendingUp, Percent, Activity, ArrowDownRight, BarChart3 } from 'lucide-react'

export function Performance() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Performance</h1>
          <p className="subheading mt-1">Analyze your portfolio returns and metrics.</p>
        </div>
        <select className="input w-auto">
          <option>Last 30 Days</option>
          <option>Last 90 Days</option>
          <option>YTD</option>
          <option>All Time</option>
        </select>
      </div>

      {/* Main Stats */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="stat-card">
          <div className="flex items-center justify-between mb-2">
            <span className="stat-label">Total Profit</span>
            <div className="icon-box icon-box-green">
              <TrendingUp className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-green">+$123,456</p>
          <p className="text-sm text-muted-foreground mt-2">Realized + Unrealized</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between mb-2">
            <span className="stat-label">Annualized Return</span>
            <div className="icon-box icon-box-blue">
              <Percent className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-blue">+12.5%</p>
          <p className="text-sm text-muted-foreground mt-2">CAGR</p>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="card p-4 flex items-center gap-3">
          <span className="dot dot-blue" />
          <div>
            <p className="text-xs text-muted-foreground">Sharpe</p>
            <p className="text-xl font-semibold">1.85</p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className="dot dot-red" />
          <div>
            <p className="text-xs text-muted-foreground">Max DD</p>
            <p className="text-xl font-semibold text-red">-8.2%</p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className="dot dot-purple" />
          <div>
            <p className="text-xs text-muted-foreground">Sortino</p>
            <p className="text-xl font-semibold">2.15</p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <span className="dot dot-green" />
          <div>
            <p className="text-xs text-muted-foreground">Win Rate</p>
            <p className="text-xl font-semibold text-green">58.3%</p>
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
              Risk Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <MetricRow label="Volatility" value="15.2%" />
            <MetricRow label="Beta" value="0.85" />
            <MetricRow label="Alpha" value="4.2%" highlight />
            <MetricRow label="Sortino Ratio" value="2.15" />
            <MetricRow label="Calmar Ratio" value="1.52" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ArrowDownRight className="h-4 w-4 text-blue" />
              Trade Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <MetricRow label="Total Trades" value="156" />
            <MetricRow label="Win Rate" value="58.3%" highlight />
            <MetricRow label="Avg Win" value="+$2,345" />
            <MetricRow label="Avg Loss" value="-$1,234" />
            <MetricRow label="Profit Factor" value="1.82" />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`flex items-center justify-between px-3 py-2.5 rounded-lg ${
      highlight ? 'bg-blue-50' : 'hover:bg-secondary'
    }`}>
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={`font-semibold mono ${highlight ? 'text-blue' : ''}`}>{value}</span>
    </div>
  )
}
