import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Plus, Edit2, ToggleLeft, ToggleRight, Brain, CheckCircle } from 'lucide-react'

const factors = [
  { name: 'MA5', expression: 'Mean($close, 5)', ic: 0.035, enabled: true },
  { name: 'MA20', expression: 'Mean($close, 20)', ic: 0.042, enabled: true },
  { name: 'RSI14', expression: 'RSI($close, 14)', ic: 0.028, enabled: true },
  { name: 'MACD', expression: 'MACD($close, 12, 26, 9)', ic: 0.031, enabled: false },
  { name: 'VOL_RATIO', expression: 'Ref($volume, 0) / Mean($volume, 20)', ic: 0.019, enabled: true },
]

export function Factors() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Factor Management</h1>
          <p className="subheading mt-1">Manage your alpha factors and expressions.</p>
        </div>
        <button className="btn btn-primary">
          <Plus className="h-4 w-4" />
          Add Factor
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Factors</span>
            <div className="icon-box icon-box-blue">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{factors.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Enabled</span>
            <div className="icon-box icon-box-green">
              <CheckCircle className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-green">{factors.filter(f => f.enabled).length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Average IC</span>
            <div className="icon-box icon-box-purple">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">
            {(factors.reduce((sum, f) => sum + f.ic, 0) / factors.length).toFixed(3)}
          </p>
        </div>
      </div>

      {/* Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Factor List</CardTitle>
          <span className="badge badge-blue">{factors.length} factors</span>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border bg-secondary/50">
                  <th className="table-header px-5 py-3 text-left">Name</th>
                  <th className="table-header px-5 py-3 text-left">Expression</th>
                  <th className="table-header px-5 py-3 text-right">IC</th>
                  <th className="table-header px-5 py-3 text-center">Status</th>
                  <th className="table-header px-5 py-3 text-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {factors.map((factor) => (
                  <tr key={factor.name} className="table-row">
                    <td className="table-cell px-5">
                      <span className="font-semibold">{factor.name}</span>
                    </td>
                    <td className="table-cell px-5">
                      <code className="text-xs mono px-2 py-1 rounded bg-secondary text-purple">
                        {factor.expression}
                      </code>
                    </td>
                    <td className="table-cell px-5 text-right">
                      <span className={`mono font-semibold ${factor.ic >= 0.03 ? 'text-green' : 'text-muted-foreground'}`}>
                        {factor.ic.toFixed(3)}
                      </span>
                    </td>
                    <td className="table-cell px-5 text-center">
                      {factor.enabled ? (
                        <span className="badge badge-green">Enabled</span>
                      ) : (
                        <span className="badge badge-gray">Disabled</span>
                      )}
                    </td>
                    <td className="table-cell px-5">
                      <div className="flex items-center justify-center gap-2">
                        <button className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors">
                          <Edit2 className="h-4 w-4" />
                        </button>
                        <button className="p-2 rounded-lg hover:bg-secondary transition-colors">
                          {factor.enabled ? (
                            <ToggleRight className="h-4 w-4 text-green" />
                          ) : (
                            <ToggleLeft className="h-4 w-4 text-muted-foreground" />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
