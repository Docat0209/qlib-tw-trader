import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Target, Loader2, Play, TrendingUp, AlertCircle, Calendar } from 'lucide-react'
import { portfolioApi, modelApi, ModelHistoryItem, PredictionsResponse } from '@/api/client'

export function Predictions() {
  const [models, setModels] = useState<ModelHistoryItem[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string>('')
  const [topK, setTopK] = useState(10)
  const [tradeDate, setTradeDate] = useState<string>('')  // 空字串 = 使用最新資料的下一天
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const parseModelId = (id: string): number => {
    if (id.startsWith('m')) {
      return parseInt(id.slice(1), 10)
    }
    return parseInt(id, 10)
  }

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await modelApi.history(20)
      // 只顯示已完成的模型，按名稱降序（最新在前）
      const completedModels = res.items
        .filter((m) => m.status === 'completed')
        .sort((a, b) => (b.name || '').localeCompare(a.name || ''))
      setModels(completedModels)
      if (completedModels.length > 0) {
        setSelectedModelId(completedModels[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  const handleGenerate = async () => {
    if (!selectedModelId) {
      setError('Please select a model')
      return
    }

    setGenerating(true)
    setError(null)
    try {
      const result = await portfolioApi.generatePredictions({
        model_id: parseModelId(selectedModelId),
        top_k: topK,
        trade_date: tradeDate || undefined,
      })
      setPredictions(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate predictions')
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Predictions</h1>
          <p className="subheading mt-1">Generate stock predictions using trained models.</p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-red-50 border border-red-100">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red" />
            <p className="text-sm text-red">{error}</p>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Target className="h-4 w-4 text-purple" />
            Generate Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 items-end">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Model</label>
              <select
                className="input w-full text-sm"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                disabled={generating}
              >
                {models.length === 0 ? (
                  <option value="">No models available</option>
                ) : (
                  models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name || m.id} - IC: {m.metrics.ic?.toFixed(4) || '---'}
                    </option>
                  ))
                )}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">
                <span className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  Trade Date (empty = next day)
                </span>
              </label>
              <input
                type="date"
                className="input w-full text-sm"
                value={tradeDate}
                onChange={(e) => setTradeDate(e.target.value)}
                disabled={generating}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Top K</label>
              <input
                type="number"
                className="input w-full text-sm"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={1}
                max={100}
                disabled={generating}
              />
            </div>
            <button
              onClick={handleGenerate}
              disabled={generating || !selectedModelId}
              className="btn btn-primary text-sm disabled:opacity-50"
            >
              {generating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Generate
                </>
              )}
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <TrendingUp className="h-4 w-4 text-green" />
            Top Predictions
          </CardTitle>
          {predictions && (
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>
                Model: <span className="font-semibold text-foreground">{predictions.model_name}</span>
              </span>
              <span>
                Trade: <span className="font-semibold text-foreground">{predictions.trade_date}</span>
              </span>
              <span>
                Data: <span className="font-semibold text-foreground">{predictions.feature_date}</span>
              </span>
            </div>
          )}
        </CardHeader>
        <CardContent className="p-0">
          {!predictions ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="icon-box icon-box-purple w-12 h-12 mb-4">
                <Target className="h-6 w-6" />
              </div>
              <p className="font-semibold">No Predictions</p>
              <p className="text-sm text-muted-foreground mt-1">Select a model and click Generate</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Rank</th>
                    <th className="table-header px-5 py-3 text-left">Symbol</th>
                    <th className="table-header px-5 py-3 text-right">Score</th>
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
                          {sig.name && <p className="text-xs text-muted-foreground">{sig.name}</p>}
                        </div>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="mono font-semibold text-green">{sig.score.toFixed(6)}</span>
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
