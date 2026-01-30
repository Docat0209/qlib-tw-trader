import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { systemApi, type DataStatusResponse } from '@/api/client'
import { Database, CheckCircle, AlertCircle, RefreshCw, Loader2, Clock } from 'lucide-react'

export function DataStatus() {
  const [data, setData] = useState<DataStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await systemApi.dataStatus()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  const freshCount = data?.stocks.filter(s => s.is_fresh).length || 0
  const totalCount = data?.stocks.length || 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Data Status</h1>
          <p className="subheading mt-1">Monitor data freshness across all datasets.</p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="btn btn-secondary disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
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

      {/* Dataset Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4 text-blue" />
            Dataset Status
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border bg-secondary/50">
                  <th className="table-header px-5 py-3 text-left">Dataset</th>
                  <th className="table-header px-5 py-3 text-left">Earliest</th>
                  <th className="table-header px-5 py-3 text-left">Latest</th>
                  <th className="table-header px-5 py-3 text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {data?.datasets.map((dataset) => (
                  <tr key={dataset.name} className="table-row">
                    <td className="table-cell px-5">
                      <span className="font-semibold">{dataset.name}</span>
                    </td>
                    <td className="table-cell px-5">
                      <span className="mono text-sm text-muted-foreground">
                        {dataset.earliest_date || '---'}
                      </span>
                    </td>
                    <td className="table-cell px-5">
                      <span className="mono text-sm text-muted-foreground">
                        {dataset.latest_date || '---'}
                      </span>
                    </td>
                    <td className="table-cell px-5 text-center">
                      {dataset.is_fresh ? (
                        <span className="badge badge-green">
                          <CheckCircle className="h-3 w-3" />
                          Fresh
                        </span>
                      ) : (
                        <span className="badge badge-red">
                          <Clock className="h-3 w-3" />
                          Stale
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Stock List */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4 text-purple" />
            Stock List
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              {freshCount}/{totalCount} fresh
            </span>
            {freshCount === totalCount ? (
              <span className="badge badge-green">All Fresh</span>
            ) : (
              <span className="badge badge-orange">{totalCount - freshCount} Stale</span>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {!data || data.stocks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8">
              <Database className="h-12 w-12 text-muted-foreground opacity-50 mb-4" />
              <p className="text-muted-foreground">No stocks found</p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {data.stocks.map((stock) => (
                <span
                  key={stock.stock_id}
                  className={`px-3 py-1.5 rounded-lg text-sm font-mono ${
                    stock.is_fresh
                      ? 'bg-green-50 text-green border border-green-200'
                      : 'bg-red-50 text-red border border-red-200'
                  }`}
                >
                  {stock.stock_id}
                </span>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Checked At */}
      {data && (
        <p className="text-xs text-muted-foreground text-center">
          Last checked: {new Date(data.checked_at).toLocaleString('en-US')}
        </p>
      )}
    </div>
  )
}
