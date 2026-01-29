import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Play, Clock, BarChart3, Layers, CheckCircle, Activity } from 'lucide-react'

export function Training() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Training Records</h1>
          <p className="subheading mt-1">View and manage model training history.</p>
        </div>
        <button className="btn btn-primary">
          <Play className="h-4 w-4" />
          Start Training
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Runs</span>
            <div className="icon-box icon-box-blue">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">0</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Success Rate</span>
            <div className="icon-box icon-box-green">
              <CheckCircle className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-muted-foreground">---</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Best IC</span>
            <div className="icon-box icon-box-purple">
              <Activity className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-muted-foreground">---</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Last Run</span>
            <div className="icon-box icon-box-orange">
              <Clock className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-sm text-muted-foreground">Never</p>
        </div>
      </div>

      {/* Training History */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue" />
            Training History
          </CardTitle>
          <span className="badge badge-gray">0 records</span>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-secondary/50">
                <th className="table-header px-5 py-3 text-left">ID</th>
                <th className="table-header px-5 py-3 text-left">Start</th>
                <th className="table-header px-5 py-3 text-left">End</th>
                <th className="table-header px-5 py-3 text-right">Model IC</th>
                <th className="table-header px-5 py-3 text-right">Factors</th>
                <th className="table-header px-5 py-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={6} className="px-5 py-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                      <BarChart3 className="h-6 w-6" />
                    </div>
                    <p className="font-semibold">No Training Records</p>
                    <p className="text-sm text-muted-foreground mt-1">Click "Start Training" to begin</p>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Training Config */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue" />
            Next Training Config
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Train Period</p>
              <p className="font-semibold mono">2020.01.01 - 2025.12.31</p>
            </div>
            <div className="p-4 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Valid Period</p>
              <p className="font-semibold mono">2026.01.01 - 2026.01.29</p>
            </div>
            <div className="p-4 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Factor Pool</p>
              <p className="font-semibold">5 factors</p>
            </div>
            <div className="p-4 rounded-lg bg-secondary">
              <p className="text-xs text-muted-foreground mb-1">Selection</p>
              <p className="font-semibold">IC Incremental</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
