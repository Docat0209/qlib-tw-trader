import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { systemApi, type DataStatusResponse } from '@/api/client'
import { Database, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react'

export function DataStatus() {
  const [data, setData] = useState<DataStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stockId, setStockId] = useState('2330')

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

  useEffect(() => {
    fetchData()
  }, [stockId])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">資料狀態</h1>
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          重新整理
        </button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            股票資料完整度
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <label className="text-sm font-medium">股票代碼</label>
            <input
              type="text"
              value={stockId}
              onChange={(e) => setStockId(e.target.value)}
              onBlur={fetchData}
              className="ml-2 rounded-md border px-3 py-1 text-sm"
              placeholder="2330"
            />
          </div>

          {error && (
            <div className="mb-4 rounded-lg bg-destructive/10 p-4 text-destructive">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : data ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="px-4 py-3 text-left font-medium">資料集</th>
                    <th className="px-4 py-3 text-left font-medium">最新日期</th>
                    <th className="px-4 py-3 text-right font-medium">筆數</th>
                    <th className="px-4 py-3 text-center font-medium">狀態</th>
                  </tr>
                </thead>
                <tbody>
                  {data.datasets.map((dataset) => (
                    <tr key={dataset.name} className="border-b">
                      <td className="px-4 py-3 font-medium">{dataset.name}</td>
                      <td className="px-4 py-3">
                        {dataset.latest_date || '-'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {dataset.record_count.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-center">
                        {dataset.latest_date ? (
                          <CheckCircle className="inline h-5 w-5 text-green-500" />
                        ) : (
                          <AlertCircle className="inline h-5 w-5 text-yellow-500" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-4 text-xs text-muted-foreground">
                檢查時間: {new Date(data.checked_at).toLocaleString('zh-TW')}
              </p>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  )
}
