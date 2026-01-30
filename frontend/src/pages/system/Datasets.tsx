import { useState, useEffect } from 'react'
import { datasetsApi, universeApi, syncApi, DatasetInfo, TestResult, CategoryInfo, StockInfo, SyncStatusResponse, MonthlyStatusResponse } from '@/api/client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Loader2, CheckCircle, XCircle, Play, ChevronDown, ChevronRight, RefreshCw, Wrench, Download } from 'lucide-react'

const statusColors: Record<string, string> = {
  available: 'bg-green-500',
  needs_accumulation: 'bg-yellow-500',
  not_implemented: 'bg-gray-400',
  pending: 'bg-slate-400',
}

const statusLabels: Record<string, string> = {
  available: '可用',
  needs_accumulation: '需累積',
  not_implemented: '未實作',
  pending: '待定',
}

const categoryLabels: Record<string, string> = {
  technical: '技術面',
  chips: '籌碼面',
  fundamental: '基本面',
  derivatives: '衍生品',
  macro: '總經指標',
}

// 檢查日期是否過時（小於昨天）
const isDateStale = (dateStr: string | null): boolean => {
  if (!dateStr) return true
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const checkDate = new Date(dateStr)
  return checkDate < yesterday
}

// 取得覆蓋率狀態顏色
// Green: 覆蓋率 >= 95% 且資料新鮮
// Yellow: 有資料但 (覆蓋率 < 95% OR 資料過時)
// Gray: 無資料
const getCoverageBarColor = (coveragePct: number, latestDate: string | null): string => {
  if (coveragePct === 0 || !latestDate) return 'bg-gray-300'
  const isComplete = coveragePct >= 95 && !isDateStale(latestDate)
  return isComplete ? 'bg-green-500' : 'bg-yellow-500'
}

const getCoverageTextColor = (coveragePct: number, latestDate: string | null): string => {
  if (coveragePct === 0 || !latestDate) return 'text-gray-400'
  const isComplete = coveragePct >= 95 && !isDateStale(latestDate)
  return isComplete ? 'text-green-600' : 'text-yellow-600'
}

// 從各種錯誤類型提取錯誤訊息
const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    // Axios/fetch 錯誤可能有 response.data.detail
    const axiosError = error as Error & { response?: { data?: { detail?: string } } }
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail
    }
    return error.message
  }
  if (typeof error === 'string') {
    return error
  }
  return '未知錯誤'
}

// 檢查 sync all 回應是否有錯誤並顯示
interface SyncAllResult {
  stocks: number
  total_inserted: number
  errors: { stock_id: string; error: string }[]
}

const checkSyncResponse = (result: SyncAllResult, datasetName: string): void => {
  if (result.errors && result.errors.length > 0) {
    // 取前 3 個錯誤訊息
    const errorMsgs = result.errors.slice(0, 3).map(e => `${e.stock_id}: ${e.error}`).join('\n')
    const moreCount = result.errors.length > 3 ? `\n...還有 ${result.errors.length - 3} 個錯誤` : ''
    alert(`${datasetName} 修復完成，但有 ${result.errors.length} 個錯誤:\n${errorMsgs}${moreCount}`)
  } else if (result.total_inserted === 0) {
    console.log(`${datasetName}: 資料已是最新，無需修復`)
  }
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
  const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null)
  const [perSyncStatus, setPerSyncStatus] = useState<SyncStatusResponse | null>(null)
  const [instSyncStatus, setInstSyncStatus] = useState<SyncStatusResponse | null>(null)
  const [marginSyncStatus, setMarginSyncStatus] = useState<SyncStatusResponse | null>(null)
  const [adjSyncStatus, setAdjSyncStatus] = useState<SyncStatusResponse | null>(null)
  const [shareholdingStatus, setShareholdingStatus] = useState<SyncStatusResponse | null>(null)
  const [securitiesLendingStatus, setSecuritiesLendingStatus] = useState<SyncStatusResponse | null>(null)
  const [monthlyRevenueStatus, setMonthlyRevenueStatus] = useState<MonthlyStatusResponse | null>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [
        datasetsRes, categoriesRes, universeRes,
        syncStatusRes, perStatusRes, instStatusRes, marginStatusRes, adjStatusRes,
        shareholdingRes, securitiesLendingRes, monthlyRevenueRes
      ] = await Promise.all([
        datasetsApi.list(),
        datasetsApi.categories(),
        universeApi.get(),
        syncApi.status('2020-01-01'),
        syncApi.perStatus('2020-01-01'),
        syncApi.institutionalStatus('2020-01-01'),
        syncApi.marginStatus('2020-01-01'),
        syncApi.adjStatus('2020-01-01'),
        syncApi.shareholdingStatus('2020-01-01'),
        syncApi.securitiesLendingStatus('2020-01-01'),
        syncApi.monthlyRevenueStatus(2020),
      ])
      setDatasets(datasetsRes.datasets)
      setCategories(categoriesRes.categories)
      setUniverse(universeRes.stocks)
      setUniverseUpdatedAt(universeRes.updated_at)
      setSyncStatus(syncStatusRes)
      setPerSyncStatus(perStatusRes)
      setInstSyncStatus(instStatusRes)
      setMarginSyncStatus(marginStatusRes)
      setAdjSyncStatus(adjStatusRes)
      setShareholdingStatus(shareholdingRes)
      setSecuritiesLendingStatus(securitiesLendingRes)
      setMonthlyRevenueStatus(monthlyRevenueRes)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const refreshSyncStatus = async () => {
    try {
      const [
        syncStatusRes, perStatusRes, instStatusRes, marginStatusRes, adjStatusRes,
        shareholdingRes, securitiesLendingRes, monthlyRevenueRes
      ] = await Promise.all([
        syncApi.status('2020-01-01'),
        syncApi.perStatus('2020-01-01'),
        syncApi.institutionalStatus('2020-01-01'),
        syncApi.marginStatus('2020-01-01'),
        syncApi.adjStatus('2020-01-01'),
        syncApi.shareholdingStatus('2020-01-01'),
        syncApi.securitiesLendingStatus('2020-01-01'),
        syncApi.monthlyRevenueStatus(2020),
      ])
      setSyncStatus(syncStatusRes)
      setPerSyncStatus(perStatusRes)
      setInstSyncStatus(instStatusRes)
      setMarginSyncStatus(marginStatusRes)
      setAdjSyncStatus(adjStatusRes)
      setShareholdingStatus(shareholdingRes)
      setSecuritiesLendingStatus(securitiesLendingRes)
      setMonthlyRevenueStatus(monthlyRevenueRes)
    } catch (error) {
      console.error('Failed to refresh sync status:', error)
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

  // 分離 pending 和 active datasets
  const pendingDatasets = datasets.filter(ds => ds.status === 'pending')
  const activeDatasets = datasets.filter(ds => ds.status !== 'pending')

  const groupedDatasets = activeDatasets.reduce((acc, ds) => {
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
                        syncStatus={syncStatus}
                        perSyncStatus={perSyncStatus}
                        instSyncStatus={instSyncStatus}
                        marginSyncStatus={marginSyncStatus}
                        adjSyncStatus={adjSyncStatus}
                        shareholdingStatus={shareholdingStatus}
                        securitiesLendingStatus={securitiesLendingStatus}
                        monthlyRevenueStatus={monthlyRevenueStatus}
                        onSyncStatusRefresh={refreshSyncStatus}
                      />
                    ))}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
      </div>

      {/* 待定 Dataset */}
      {pendingDatasets.length > 0 && (
        <Card className="border-slate-300 bg-slate-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg text-slate-600">
              待定 Dataset
              <span className="ml-2 px-2 py-0.5 text-xs rounded bg-slate-400 text-white">
                {pendingDatasets.length}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-slate-500 mb-3">
              以下資料集尚未納入核心因子，僅作紀錄參考
            </div>
            <div className="space-y-2">
              {pendingDatasets.map((ds) => (
                <div key={ds.name} className="border rounded-lg p-3 bg-white">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 text-xs text-white rounded bg-slate-400">
                        待定
                      </span>
                      <div>
                        <div className="font-medium">{ds.display_name}</div>
                        <div className="text-sm text-muted-foreground font-mono">
                          {ds.name}
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {ds.source}
                    </div>
                  </div>
                  {ds.description && (
                    <div className="mt-2 text-xs text-slate-500">
                      {ds.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

interface DatasetRowProps {
  dataset: DatasetInfo
  testResult?: TestResult
  isTesting: boolean
  onTest: () => void
  syncStatus: SyncStatusResponse | null
  perSyncStatus: SyncStatusResponse | null
  instSyncStatus: SyncStatusResponse | null
  marginSyncStatus: SyncStatusResponse | null
  adjSyncStatus: SyncStatusResponse | null
  shareholdingStatus: SyncStatusResponse | null
  securitiesLendingStatus: SyncStatusResponse | null
  monthlyRevenueStatus: MonthlyStatusResponse | null
  onSyncStatusRefresh: () => void
}

function DatasetRow({ dataset, testResult, isTesting, onTest, syncStatus, perSyncStatus, instSyncStatus, marginSyncStatus, adjSyncStatus, shareholdingStatus, securitiesLendingStatus, monthlyRevenueStatus, onSyncStatusRefresh }: DatasetRowProps) {
  const [expanded, setExpanded] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [repairing, setRepairing] = useState(false)

  // TaiwanStockPrice 特殊處理
  if (dataset.name === 'TaiwanStockPrice') {
    const totalStocks = syncStatus?.stocks.length || 0
    const completeStocks = syncStatus?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = syncStatus?.trading_days || 0

    // 找出最早和最晚日期
    const allEarliest = syncStatus?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = syncStatus?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.bulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        // 先同步交易日曆
        await syncApi.calendar('2020-01-01')
        // 再同步所有股票
        const result = await syncApi.all('2020-01-01')
        checkSyncResponse(result, '日K線')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-blue-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-blue-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 按鈕組 */}
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (TWSE)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        {/* 狀態面板 */}
        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {/* 股票覆蓋率預覽 */}
        {syncStatus && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-blue-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {syncStatus.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {/* 展開詳情 */}
        {expanded && syncStatus && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {syncStatus.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockPER 特殊處理
  if (dataset.name === 'TaiwanStockPER') {
    const status = perSyncStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    // 找出最早和最晚日期
    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.perBulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        // 先同步交易日曆
        await syncApi.calendar('2020-01-01')
        // 再同步所有股票
        const result = await syncApi.perAll('2020-01-01')
        checkSyncResponse(result, 'PER/PBR/殖利率')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-green-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-green-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 按鈕組 */}
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (TWSE)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        {/* 狀態面板 */}
        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {/* 股票覆蓋率預覽 */}
        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-green-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {/* 展開詳情 */}
        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockInstitutionalInvestorsBuySell 特殊處理
  if (dataset.name === 'TaiwanStockInstitutionalInvestorsBuySell') {
    const status = instSyncStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    // 找出最早和最晚日期
    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.institutionalBulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        // 先同步交易日曆
        await syncApi.calendar('2020-01-01')
        // 再同步所有股票
        const result = await syncApi.institutionalAll('2020-01-01')
        checkSyncResponse(result, '三大法人')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-purple-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-purple-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 按鈕組 */}
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (TWSE)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        {/* 狀態面板 */}
        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {/* 股票覆蓋率預覽 */}
        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-purple-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {/* 展開詳情 */}
        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockPriceAdj 特殊處理 (還原股價)
  if (dataset.name === 'TaiwanStockPriceAdj') {
    const status = adjSyncStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    // 找出最早和最晚日期
    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.adjBulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        // 先同步交易日曆
        await syncApi.calendar('2020-01-01')
        // 再同步所有股票
        const result = await syncApi.adjAll('2020-01-01')
        checkSyncResponse(result, '還原股價')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-cyan-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-cyan-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 按鈕組 */}
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-cyan-500 text-white rounded hover:bg-cyan-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (yfinance)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (yfinance)</span>
            </button>
          </div>
        </div>

        {/* 狀態面板 */}
        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {/* 股票覆蓋率預覽 */}
        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-cyan-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {/* 展開詳情 */}
        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockMarginPurchaseShortSale 特殊處理
  if (dataset.name === 'TaiwanStockMarginPurchaseShortSale') {
    const status = marginSyncStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    // 找出最早和最晚日期
    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.marginBulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        // 先同步交易日曆
        await syncApi.calendar('2020-01-01')
        // 再同步所有股票
        const result = await syncApi.marginAll('2020-01-01')
        checkSyncResponse(result, '融資融券')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-amber-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-amber-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 按鈕組 */}
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (TWSE)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        {/* 狀態面板 */}
        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {/* 股票覆蓋率預覽 */}
        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-amber-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {/* 展開詳情 */}
        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockShareholding 特殊處理 (外資持股)
  if (dataset.name === 'TaiwanStockShareholding') {
    const status = shareholdingStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleSync = async () => {
      setSyncing(true)
      try {
        await syncApi.shareholdingBulk()
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Sync failed:', error)
        alert(`同步失敗: ${getErrorMessage(error)}`)
      } finally {
        setSyncing(false)
      }
    }

    const handleRepair = async () => {
      setRepairing(true)
      try {
        await syncApi.calendar('2020-01-01')
        const result = await syncApi.shareholdingAll('2020-01-01')
        checkSyncResponse(result, '外資持股')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-indigo-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-indigo-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-indigo-500 text-white rounded hover:bg-indigo-600 disabled:opacity-50"
              onClick={handleSync}
              disabled={syncing || repairing}
            >
              {syncing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              <span>Sync (TWSE)</span>
            </button>
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={syncing || repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-indigo-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockSecuritiesLending 特殊處理 (借券明細)
  if (dataset.name === 'TaiwanStockSecuritiesLending') {
    const status = securitiesLendingStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
    const tradingDays = status?.trading_days || 0

    const allEarliest = status?.stocks
      .filter(s => s.earliest_date)
      .map(s => s.earliest_date!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_date)
      .map(s => s.latest_date!)
      .sort()
      .reverse() || []

    const earliestDate = allEarliest[0] || '無資料'
    const latestDate = allLatest[0] || '無資料'

    const handleRepair = async () => {
      setRepairing(true)
      try {
        await syncApi.calendar('2020-01-01')
        const result = await syncApi.securitiesLendingAll('2020-01-01')
        checkSyncResponse(result, '借券明細')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-pink-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-pink-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">交易日數</div>
            <div className="text-lg font-bold">{tradingDays.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestDate}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestDate}</div>
          </div>
        </div>

        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-pink-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${getCoverageBarColor(stock.coverage_pct, stock.latest_date)}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早</th>
                  <th className="text-left p-2">最新</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_date || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_date || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${getCoverageTextColor(stock.coverage_pct, stock.latest_date)}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // TaiwanStockMonthRevenue 特殊處理 (月營收)
  if (dataset.name === 'TaiwanStockMonthRevenue') {
    const status = monthlyRevenueStatus
    const totalStocks = status?.stocks.length || 0
    const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95).length || 0
    const expectedMonths = status?.expected_months || 0

    const allEarliest = status?.stocks
      .filter(s => s.earliest_month)
      .map(s => s.earliest_month!)
      .sort() || []
    const allLatest = status?.stocks
      .filter(s => s.latest_month)
      .map(s => s.latest_month!)
      .sort()
      .reverse() || []

    const earliestMonth = allEarliest[0] || '無資料'
    const latestMonth = allLatest[0] || '無資料'

    const handleRepair = async () => {
      setRepairing(true)
      try {
        const result = await syncApi.monthlyRevenueAll(2020)
        checkSyncResponse(result, '月營收')
        onSyncStatusRefresh()
      } catch (error) {
        console.error('Repair failed:', error)
        alert(`修復失敗: ${getErrorMessage(error)}`)
      } finally {
        setRepairing(false)
      }
    }

    return (
      <div className="border rounded-lg p-4 bg-emerald-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="px-2 py-0.5 text-xs text-white rounded bg-emerald-500">
              核心
            </span>
            <div>
              <div className="font-medium">{dataset.display_name}</div>
              <div className="text-sm text-muted-foreground font-mono">
                {dataset.name}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={handleRepair}
              disabled={repairing}
            >
              {repairing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              <span>修復資料 (FinMind)</span>
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-4 gap-4">
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">預期月數</div>
            <div className="text-lg font-bold">{expectedMonths}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">完整股票</div>
            <div className="text-lg font-bold">
              {completeStocks}/{totalStocks}
              <span className="text-sm font-normal text-muted-foreground ml-1">
                ({totalStocks > 0 ? Math.round(completeStocks / totalStocks * 100) : 0}%)
              </span>
            </div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最早資料</div>
            <div className="text-lg font-bold font-mono">{earliestMonth}</div>
          </div>
          <div className="p-3 bg-white rounded border">
            <div className="text-xs text-muted-foreground">最新資料</div>
            <div className="text-lg font-bold font-mono">{latestMonth}</div>
          </div>
        </div>

        {status && (
          <div className="mt-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">覆蓋率分佈</span>
              <button
                className="text-xs text-emerald-500 hover:underline"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? '收起' : '展開詳情'}
              </button>
            </div>
            <div className="flex gap-0.5">
              {status.stocks.slice(0, 100).map((stock) => (
                <div
                  key={stock.stock_id}
                  className={`w-2 h-4 rounded-sm ${stock.coverage_pct >= 95 ? 'bg-green-500' : stock.coverage_pct > 0 ? 'bg-yellow-500' : 'bg-gray-300'}`}
                  title={`${stock.stock_id} ${stock.name}: ${stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%`}
                />
              ))}
            </div>
          </div>
        )}

        {expanded && status && (
          <div className="mt-3 max-h-60 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted sticky top-0">
                <tr>
                  <th className="text-left p-2">代碼</th>
                  <th className="text-left p-2">名稱</th>
                  <th className="text-left p-2">最早月份</th>
                  <th className="text-left p-2">最新月份</th>
                  <th className="text-right p-2">筆數</th>
                  <th className="text-right p-2">覆蓋率</th>
                </tr>
              </thead>
              <tbody>
                {status.stocks.map((stock) => (
                  <tr key={stock.stock_id} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono">{stock.stock_id}</td>
                    <td className="p-2">{stock.name}</td>
                    <td className="p-2 font-mono text-xs">{stock.earliest_month || '-'}</td>
                    <td className="p-2 font-mono text-xs">{stock.latest_month || '-'}</td>
                    <td className="p-2 text-right">{stock.total_records.toLocaleString()}</td>
                    <td className={`p-2 text-right font-medium ${stock.coverage_pct >= 95 ? 'text-green-600' : stock.coverage_pct > 0 ? 'text-yellow-600' : 'text-gray-400'}`}>
                      {stock.coverage_pct >= 99 ? 100 : stock.coverage_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // 其他 dataset 的一般顯示
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
