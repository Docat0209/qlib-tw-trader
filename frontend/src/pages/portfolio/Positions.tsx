import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Briefcase, TrendingDown, DollarSign, Package } from 'lucide-react'

export function Positions() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Positions</h1>
          <p className="subheading mt-1">View and manage your current holdings.</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary">
          <span className="dot dot-orange" />
          <span className="text-sm font-medium">Market Closed</span>
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
          <p className="stat-value">$1.23M</p>
          <p className="text-sm text-muted-foreground mt-2">NAV</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Positions</span>
            <div className="icon-box icon-box-green">
              <Package className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">5</p>
          <p className="text-sm text-muted-foreground mt-2">Active holdings</p>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Today P/L</span>
            <div className="icon-box icon-box-orange">
              <TrendingDown className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-red">-$5,432</p>
          <p className="text-sm text-muted-foreground mt-2">-0.44%</p>
        </div>
      </div>

      {/* Positions Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Briefcase className="h-4 w-4 text-blue" />
            Position Details
          </CardTitle>
          <span className="badge badge-blue">5 positions</span>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-secondary/50">
                <th className="table-header px-5 py-3 text-left">Symbol</th>
                <th className="table-header px-5 py-3 text-right">Shares</th>
                <th className="table-header px-5 py-3 text-right">Cost</th>
                <th className="table-header px-5 py-3 text-right">Price</th>
                <th className="table-header px-5 py-3 text-right">P/L</th>
                <th className="table-header px-5 py-3 text-right">Return</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={6} className="px-5 py-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                      <Briefcase className="h-6 w-6" />
                    </div>
                    <p className="font-semibold">No Positions</p>
                    <p className="text-sm text-muted-foreground mt-1">Execute trades to see positions here</p>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Sector Allocation */}
      <Card>
        <CardHeader>
          <CardTitle>Sector Allocation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-3">
            {[
              { name: 'Tech', color: 'bg-blue-50 text-blue' },
              { name: 'Finance', color: 'bg-green-50 text-green' },
              { name: 'Industry', color: 'bg-orange-50 text-orange' },
              { name: 'Consumer', color: 'bg-purple-50 text-purple' },
              { name: 'Other', color: 'bg-secondary text-muted-foreground' },
            ].map((sector) => (
              <div key={sector.name} className={`p-4 rounded-lg text-center ${sector.color}`}>
                <p className="text-xs text-muted-foreground">{sector.name}</p>
                <p className="text-lg font-semibold mt-1">---%</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
