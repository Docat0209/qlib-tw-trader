import { useState, useEffect } from 'react'
import { datasetsApi, DatasetInfo, TestResult, CategoryInfo } from '@/api/client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Loader2, CheckCircle, XCircle, Play, ChevronDown, ChevronRight } from 'lucide-react'

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
  const [loading, setLoading] = useState(true)
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [testingDataset, setTestingDataset] = useState<string | null>(null)
  const [stockId, setStockId] = useState('2330')
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['technical', 'chips']))
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [datasetsRes, categoriesRes] = await Promise.all([
        datasetsApi.list(),
        datasetsApi.categories(),
      ])
      setDatasets(datasetsRes.datasets)
      setCategories(categoriesRes.categories)
    } catch (error) {
      console.error('Failed to load datasets:', error)
    } finally {
      setLoading(false)
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
