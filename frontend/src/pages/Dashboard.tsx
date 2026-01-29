import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { TrendingUp, TrendingDown, Database, Brain, DollarSign, Activity } from 'lucide-react'

export function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Good afternoon, Shane</h1>
          <p className="subheading mt-1">Here's what's happening with your portfolio today.</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="dot dot-green" />
          <span className="text-sm text-muted-foreground">Live</span>
          <span className="text-sm text-muted-foreground mx-2">|</span>
          <span className="text-sm mono">2026.01.29</span>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Return</span>
            <div className="icon-box icon-box-green">
              <TrendingUp className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-green">+12.5%</p>
          <div className="stat-change positive">
            <TrendingUp className="h-3 w-3" />
            <span>+2.1% from last month</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Portfolio Value</span>
            <div className="icon-box icon-box-blue">
              <DollarSign className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">$1.23M</p>
          <div className="stat-change positive">
            <TrendingUp className="h-3 w-3" />
            <span>5 active positions</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Today's P/L</span>
            <div className="icon-box icon-box-orange">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-red">-$5,432</p>
          <div className="stat-change negative">
            <TrendingDown className="h-3 w-3" />
            <span>-0.44%</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Active Factors</span>
            <div className="icon-box icon-box-purple">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">24</p>
          <p className="text-sm text-muted-foreground mt-2">IC: 0.045</p>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatusCard label="Data Status" value="Normal" status="green" />
        <StatusCard label="Model Version" value="v2.1" status="blue" />
        <StatusCard label="Today Trades" value="3" status="blue" />
        <StatusCard label="Risk Level" value="Medium" status="orange" />
      </div>

      {/* Content Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Recent Trades */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Trades</CardTitle>
            <span className="badge badge-blue">Today</span>
          </CardHeader>
          <CardContent className="space-y-3">
            <TradeRow action="BUY" symbol="2330" name="TSMC" shares="1000" price="580" time="14:30" />
            <TradeRow action="SELL" symbol="2454" name="MediaTek" shares="500" price="720" time="10:15" />
            <TradeRow action="BUY" symbol="2317" name="Hon Hai" shares="2000" price="105" time="09:30" />
          </CardContent>
        </Card>

        {/* System Status */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>System Status</CardTitle>
            <span className="badge badge-green">Online</span>
          </CardHeader>
          <CardContent className="space-y-2">
            <StatusRow label="Data Pipeline" value="Running" color="green" />
            <StatusRow label="Model Server" value="Ready" color="green" />
            <StatusRow label="Trade Engine" value="Standby" color="orange" />
            <StatusRow label="Last Sync" value="14:30:22" color="gray" />
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
            <ActionButton label="View Report" icon={<Activity className="h-5 w-5" />} color="green" />
            <ActionButton label="Settings" icon={<TrendingUp className="h-5 w-5" />} color="orange" />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function StatusCard({ label, value, status }: { label: string; value: string; status: string }) {
  const badgeClass = `badge-${status}`
  return (
    <div className="card p-4 flex items-center justify-between">
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="font-semibold">{value}</p>
      </div>
      <span className={`badge ${badgeClass}`}>{status === 'green' ? 'OK' : status === 'orange' ? 'WARN' : 'ACTIVE'}</span>
    </div>
  )
}

function TradeRow({
  action,
  symbol,
  name,
  shares,
  price,
  time,
}: {
  action: 'BUY' | 'SELL'
  symbol: string
  name: string
  shares: string
  price: string
  time: string
}) {
  const isBuy = action === 'BUY'
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-secondary transition-colors">
      <div className={`icon-box ${isBuy ? 'icon-box-green' : 'icon-box-orange'}`}>
        {isBuy ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`badge ${isBuy ? 'badge-green' : 'badge-red'}`}>{action}</span>
          <span className="font-semibold">{symbol}</span>
          <span className="text-sm text-muted-foreground">{name}</span>
        </div>
        <p className="text-xs text-muted-foreground mono">{shares} shares @ ${price}</p>
      </div>
      <span className="text-xs mono text-muted-foreground">{time}</span>
    </div>
  )
}

function StatusRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <span className={`dot dot-${color}`} />
        <span className="text-sm font-medium">{value}</span>
      </div>
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
