import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { systemApi, type DataStatusResponse, type SyncResponse } from '@/api/client'
import { Database, CheckCircle, AlertCircle, RefreshCw, Search, Download, Loader2 } from 'lucide-react'

export function DataStatus() {
  const [data, setData] = useState<DataStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stockId, setStockId] = useState('2330')
  const [syncing, setSyncing] = useState(false)
  const [syncResult, setSyncResult] = useState<SyncResponse | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await systemApi.dataStatus(stockId)
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  const handleSync = async () => {
    setSyncing(true)
    setSyncResult(null)
    setError(null)
    try {
      // Default: sync last 30 days
      const endDate = new Date()
      const startDate = new Date()
      startDate.setDate(startDate.getDate() - 30)

      const result = await systemApi.sync({
        stock_id: stockId,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0],
      })
      setSyncResult(result)
      // Refresh data status after sync
      await fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sync failed')
    } finally {
      setSyncing(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [stockId])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Data Status</h1>
          <p className="subheading mt-1">Check data completeness and sync missing data.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleSync}
            disabled={syncing || loading}
            className="btn btn-primary disabled:opacity-50"
          >
            {syncing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Download className="h-4 w-4" />
            )}
            {syncing ? 'Syncing...' : 'Sync Data'}
          </button>
          <button
            onClick={fetchData}
            disabled={loading || syncing}
            className="btn btn-secondary disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-5">
          <div className="flex items-center gap-6">
            <div className="icon-box icon-box-blue">
              <Search className="h-4 w-4" />
            </div>
            <div className="flex-1">
              <label className="text-xs text-muted-foreground block mb-1">Stock ID</label>
              <input
                type="text"
                value={stockId}
                onChange={(e) => setStockId(e.target.value)}
                onBlur={fetchData}
                onKeyDown={(e) => e.key === 'Enter' && fetchData()}
                className="input max-w-xs"
                placeholder="Enter ID..."
              />
            </div>
            <div className="text-right">
              <p className="text-xs text-muted-foreground mb-1">Status</p>
              <div className="flex items-center gap-2">
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 text-blue animate-spin" />
                    <span className="text-sm font-medium">Loading...</span>
                  </>
                ) : syncing ? (
                  <>
                    <Loader2 className="h-4 w-4 text-purple animate-spin" />
                    <span className="text-sm font-medium text-purple">Syncing...</span>
                  </>
                ) : error ? (
                  <>
                    <AlertCircle className="h-4 w-4 text-red" />
                    <span className="text-sm font-medium text-red">Error</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-4 w-4 text-green" />
                    <span className="text-sm font-medium text-green">Done</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-100">
          <div className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-red" />
            <div>
              <p className="font-semibold text-red">Error Occurred</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Sync Result */}
      {syncResult && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Download className="h-4 w-4 text-green" />
              Sync Complete
            </CardTitle>
            <span className="text-xs text-muted-foreground mono">
              {new Date(syncResult.synced_at).toLocaleString('en-US')}
            </span>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Total Records</p>
                <p className="text-xl font-semibold text-green">{syncResult.total_records.toLocaleString()}</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Datasets</p>
                <p className="text-xl font-semibold">{syncResult.results.length}</p>
              </div>
              <div className="p-3 rounded-lg bg-secondary">
                <p className="text-xs text-muted-foreground">Success Rate</p>
                <p className="text-xl font-semibold text-green">
                  {Math.round((syncResult.results.filter(r => r.success).length / syncResult.results.length) * 100)}%
                </p>
              </div>
            </div>
            <div className="space-y-2">
              {syncResult.results.map((result) => (
                <div
                  key={result.dataset}
                  className="flex items-center justify-between p-2 rounded-lg hover:bg-secondary"
                >
                  <div className="flex items-center gap-2">
                    {result.success ? (
                      <CheckCircle className="h-4 w-4 text-green" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red" />
                    )}
                    <span className="font-medium">{result.dataset}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground">
                      {result.records_fetched} records
                    </span>
                    {result.error && (
                      <span className="text-xs text-red">{result.error}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4 text-blue" />
            Dataset Status
          </CardTitle>
          {data && (
            <span className="text-xs text-muted-foreground mono">
              Checked: {new Date(data.checked_at).toLocaleString('en-US')}
            </span>
          )}
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <RefreshCw className="h-6 w-6 animate-spin" />
              </div>
              <p className="text-muted-foreground text-sm">Scanning datasets...</p>
            </div>
          ) : data ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Dataset</th>
                    <th className="table-header px-5 py-3 text-left">Latest Date</th>
                    <th className="table-header px-5 py-3 text-right">Records</th>
                    <th className="table-header px-5 py-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.datasets.map((dataset) => (
                    <tr key={dataset.name} className="table-row">
                      <td className="table-cell px-5">
                        <span className="font-semibold">{dataset.name}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="mono text-sm text-muted-foreground">
                          {dataset.latest_date || '---'}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="mono font-semibold">{dataset.record_count.toLocaleString()}</span>
                      </td>
                      <td className="table-cell px-5 text-center">
                        {dataset.latest_date ? (
                          <span className="badge badge-green">
                            <CheckCircle className="h-3 w-3" />
                            OK
                          </span>
                        ) : (
                          <span className="badge badge-orange">
                            <AlertCircle className="h-3 w-3" />
                            Empty
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {/* Legend */}
      <div className="flex items-center gap-6 p-4 card">
        <span className="text-xs text-muted-foreground">Legend:</span>
        <div className="flex items-center gap-2">
          <span className="dot dot-green" />
          <span className="text-sm">OK</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="dot dot-orange" />
          <span className="text-sm">Empty</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="dot dot-red" />
          <span className="text-sm">Error</span>
        </div>
      </div>
    </div>
  )
}
