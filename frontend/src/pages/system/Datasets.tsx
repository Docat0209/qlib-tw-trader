import { useState, useEffect } from 'react'
import { datasetsApi, universeApi, DatasetInfo, TestResult, CategoryInfo, StockInfo } from '@/api/client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Loader2, CheckCircle, XCircle, Play, ChevronDown, ChevronRight, RefreshCw } from 'lucide-react'

const statusColors: Record<string, string> = {
  available: 'bg-green-500',
  needs_accumulation: 'bg-yellow-500',
  not_implemented: 'bg-gray-400',
}

const statusLabels: Record<string, string> = {
  available: '可用',
  needs_accumulation: '需累積',
  not_implemented: '未實作',
}

const categoryLabels: Record<string, string> = {
  technical: '技術面',
  chips: '籌碼面',
  fundamental: '基本面',
  derivatives: '衍生品',
  macro: '總經指標',
}

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [categories, setCategories] = useState<CategoryInfo[]>([])
  const [universe, setUniverse] = useState<StockInfo[]>([])
  const [universeUpdatedAt, setUniverseUpdatedAt] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [syncingUniverse, setSyncingUniverse] = useState(false)
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [testingDataset, setTestingDataset] = useState<string | null>(null)
  const [stockId, setStockId] = useState('2330')
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['technical', 'chips']))
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [showUniverse, setShowUniverse] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [datasetsRes, categoriesRes, universeRes] = await Promise.all([
        datasetsApi.list(),
        datasetsApi.categories(),
        universeApi.get(),
      ])
      setDatasets(datasetsRes.datasets)
      setCategories(categoriesRes.categories)
      setUniverse(universeRes.stocks)
      setUniverseUpdatedAt(universeRes.updated_at)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const syncUniverse = async () => {
    setSyncingUniverse(true)
    try {
      await universeApi.sync()
      const universeRes = await universeApi.get()
      setUniverse(universeRes.stocks)
      setUniverseUpdatedAt(universeRes.updated_at)
    } catch (error) {
      console.error('Failed to sync universe:', error)
    } finally {
      setSyncingUniverse(false)
    }
  }

  const testDataset = async (datasetName: string) => {
    setTestingDataset(datasetName)
    try {
      const result = await datasetsApi.test(datasetName, stockId, 7)
      setTestResults(prev => ({ ...prev, [datasetName]: result }))
    } catch (error) {
      setTestResults(prev => ({
        ...prev,
        [datasetName]: {
          dataset: datasetName,
          success: false,
          record_count: 0,
          sample_data: null,
          error: String(error),
        },
      }))
    } finally {
      setTestingDataset(null)
    }
  }

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      return next
    })
  }

  const groupedDatasets = datasets.reduce((acc, ds) => {
    if (!acc[ds.category]) acc[ds.category] = []
    acc[ds.category].push(ds)
    return acc
  }, {} as Record<string, DatasetInfo[]>)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Datasets 資料集</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">測試股票:</label>
            <input
              type="text"
              value={stockId}
              onChange={(e) => setStockId(e.target.value)}
              className="w-24 px-2 py-1 text-sm border rounded bg-background"
              placeholder="股票代碼"
            />
          </div>
        </div>
      </div>

      {/* 股票池 */}
      <Card>
        <CardHeader
          className="cursor-pointer"
          onClick={() => setShowUniverse(!showUniverse)}
        >
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {showUniverse ? (
                <ChevronDown className="h-5 w-5" />
              ) : (
                <ChevronRight className="h-5 w-5" />
              )}
              股票池 (tw100)
              <span className="ml-2 px-2 py-0.5 text-xs rounded bg-blue-500 text-white">
                {universe.length} 檔
              </span>
            </div>
            <div className="flex items-center gap-3">
              {universeUpdatedAt && (
                <span className="text-xs text-muted-foreground">
                  更新: {new Date(universeUpdatedAt).toLocaleDateString()}
                </span>
              )}
              <button
                className="flex items-center gap-1 px-3 py-1 text-sm border rounded hover:bg-secondary disabled:opacity-50"
                onClick={(e) => {
                  e.stopPropagation()
                  syncUniverse()
                }}
                disabled={syncingUniverse}
              >
                <RefreshCw className={`h-4 w-4 ${syncingUniverse ? 'animate-spin' : ''}`} />
                <span>更新</span>
              </button>
            </div>
          </CardTitle>
        </CardHeader>
        {showUniverse && (
          <CardContent>
            <div className="text-sm text-muted-foreground mb-3">
              台股市值前 100 大（排除 ETF、KY 股）
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 max-h-[600px] overflow-y-auto">
              {universe.map((stock) => (
                <div
                  key={stock.stock_id}
                  className="border rounded-lg p-3 hover:border-primary cursor-pointer transition-colors"
                  onClick={() => setStockId(stock.stock_id)}
                >
                  {/* 股票標題 */}
                  <div className="flex items-center justify-between mb-2 pb-2 border-b">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold">{stock.stock_id}</span>
                      <span className="text-sm text-muted-foreground">{stock.name}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">#{stock.rank}</span>
                  </div>

                  {/* 資料監控面板 */}
                  <div className="grid grid-cols-2 gap-2">
                    {/* 技術面 */}
                    <div className="p-2 rounded bg-blue-500/10 border border-blue-500/20">
                      <div className="text-xs font-medium text-blue-600 mb-1">技術面</div>
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="日K線"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="還原價"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="PER"></span>
                      </div>
                    </div>

                    {/* 籌碼面 */}
                    <div className="p-2 rounded bg-green-500/10 border border-green-500/20">
                      <div className="text-xs font-medium text-green-600 mb-1">籌碼面</div>
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="法人"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="融資券"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="外資"></span>
                      </div>
                    </div>

                    {/* 基本面 */}
                    <div className="p-2 rounded bg-orange-500/10 border border-orange-500/20">
                      <div className="text-xs font-medium text-orange-600 mb-1">基本面</div>
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="營收"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="財報"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="股利"></span>
                      </div>
                    </div>

                    {/* 衍生品 */}
                    <div className="p-2 rounded bg-purple-500/10 border border-purple-500/20">
                      <div className="text-xs font-medium text-purple-600 mb-1">衍生品</div>
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="期貨"></span>
                        <span className="w-2 h-2 rounded-full bg-gray-300" title="選擇權"></span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        )}
      </Card>

      {/* 類別摘要 */}
      <div className="grid grid-cols-5 gap-4">
        {categories.map((cat) => (
          <Card
            key={cat.id}
            className={`cursor-pointer transition-colors hover:border-primary ${
              selectedCategory === cat.id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedCategory(selectedCategory === cat.id ? null : cat.id)}
          >
            <CardContent className="pt-4">
              <div className="text-sm text-muted-foreground">{cat.name}</div>
              <div className="text-2xl font-bold">
                {cat.available}/{cat.total}
              </div>
              <div className="text-xs text-muted-foreground">可用/總數</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* 資料集列表 */}
      <div className="space-y-4">
        {Object.entries(groupedDatasets)
          .filter(([category]) => !selectedCategory || category === selectedCategory)
          .map(([category, items]) => (
            <Card key={category}>
              <CardHeader
                className="cursor-pointer"
                onClick={() => toggleCategory(category)}
              >
                <CardTitle className="flex items-center gap-2 text-lg">
                  {expandedCategories.has(category) ? (
                    <ChevronDown className="h-5 w-5" />
                  ) : (
                    <ChevronRight className="h-5 w-5" />
                  )}
                  {categoryLabels[category] || category}
                  <span className="ml-2 px-2 py-0.5 text-xs rounded bg-secondary">
                    {items.length}
                  </span>
                </CardTitle>
              </CardHeader>
              {expandedCategories.has(category) && (
                <CardContent>
                  <div className="space-y-2">
                    {items.map((ds) => (
                      <DatasetRow
                        key={ds.name}
                        dataset={ds}
                        testResult={testResults[ds.name]}
                        isTesting={testingDataset === ds.name}
                        onTest={() => testDataset(ds.name)}
                      />
                    ))}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
      </div>
    </div>
  )
}

interface DatasetRowProps {
  dataset: DatasetInfo
  testResult?: TestResult
  isTesting: boolean
  onTest: () => void
}

function DatasetRow({ dataset, testResult, isTesting, onTest }: DatasetRowProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border rounded-lg p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`px-2 py-0.5 text-xs text-white rounded ${statusColors[dataset.status]}`}>
            {statusLabels[dataset.status]}
          </span>
          <div>
            <div className="font-medium">{dataset.display_name}</div>
            <div className="text-sm text-muted-foreground font-mono">
              {dataset.name}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-sm text-muted-foreground">
            {dataset.source}
          </div>
          {dataset.status === 'available' && (
            <button
              className="flex items-center gap-1 px-3 py-1 text-sm border rounded hover:bg-secondary disabled:opacity-50"
              onClick={onTest}
              disabled={isTesting}
            >
              {isTesting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>測試</span>
            </button>
          )}
          {testResult && (
            <div className="flex items-center gap-2">
              {testResult.success ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-600">
                    {testResult.record_count} 筆
                  </span>
                </>
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              {testResult.sample_data && (
                <button
                  className="px-2 py-0.5 text-xs hover:bg-secondary rounded"
                  onClick={() => setExpanded(!expanded)}
                >
                  {expanded ? '收起' : '查看'}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {dataset.description && (
        <div className="mt-1 text-xs text-muted-foreground">
          {dataset.description}
        </div>
      )}

      {testResult?.error && (
        <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
          {testResult.error}
        </div>
      )}

      {expanded && testResult?.sample_data && (
        <div className="mt-3 overflow-auto">
          <pre className="text-xs bg-muted p-2 rounded">
            {JSON.stringify(testResult.sample_data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
