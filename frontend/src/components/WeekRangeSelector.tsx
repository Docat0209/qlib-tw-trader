import { cn } from '@/lib/utils'
import { WeekStatus } from '@/api/client'
import { CheckCircle, AlertTriangle, XCircle, ArrowRight } from 'lucide-react'

interface WeekRangeSelectorProps {
  weeks: WeekStatus[]
  startWeek: string | null
  endWeek: string | null
  onRangeChange: (start: string | null, end: string | null) => void
}

export function WeekRangeSelector({
  weeks,
  startWeek,
  endWeek,
  onRangeChange,
}: WeekRangeSelectorProps) {
  // 按年份分組
  const byYear = weeks.reduce((acc, week) => {
    const year = week.week_id.slice(0, 4)
    if (!acc[year]) acc[year] = []
    acc[year].push(week)
    return acc
  }, {} as Record<string, WeekStatus[]>)

  // 每年內按週數排序（降序）
  Object.values(byYear).forEach(yearWeeks => {
    yearWeeks.sort((a, b) => b.week_id.localeCompare(a.week_id))
  })

  // 按年份降序排列
  const sortedYears = Object.keys(byYear).sort((a, b) => b.localeCompare(a))

  // 點擊處理
  const handleClick = (weekId: string, status: string) => {
    if (status === 'not_allowed') return

    if (!startWeek) {
      // 沒有起始週，設定起始週
      onRangeChange(weekId, null)
    } else if (!endWeek) {
      // 有起始週但沒有結束週
      if (weekId === startWeek) {
        // 點擊同一週，清除
        onRangeChange(null, null)
      } else if (weekId < startWeek) {
        // 選的週比起始週早，交換
        onRangeChange(weekId, startWeek)
      } else {
        // 設定結束週
        onRangeChange(startWeek, weekId)
      }
    } else {
      // 已有完整範圍，重新開始選擇
      onRangeChange(weekId, null)
    }
  }

  // 判斷週是否在選擇範圍內
  const isInRange = (weekId: string) => {
    if (!startWeek) return false
    if (!endWeek) return weekId === startWeek
    return weekId >= startWeek && weekId <= endWeek
  }

  return (
    <div className="space-y-4">
      {/* 選擇顯示 */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Range:</span>
        {startWeek ? (
          <>
            <span className="font-mono font-medium text-blue-600">{startWeek}</span>
            <ArrowRight className="h-3 w-3 text-muted-foreground" />
            <span className={cn(
              "font-mono font-medium",
              endWeek ? "text-blue-600" : "text-muted-foreground"
            )}>
              {endWeek || '(select end)'}
            </span>
          </>
        ) : (
          <span className="text-muted-foreground">(select start week)</span>
        )}
        {(startWeek || endWeek) && (
          <button
            onClick={() => onRangeChange(null, null)}
            className="text-xs text-red-500 hover:text-red-600 ml-2"
          >
            Clear
          </button>
        )}
      </div>

      {/* 週曆 */}
      {sortedYears.map((year) => {
        const yearWeeks = byYear[year]

        return (
          <div key={year}>
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-sm text-muted-foreground">{year}</h3>
              <span className="text-[10px] text-muted-foreground">
                {yearWeeks.filter(w => w.status === 'available').length} available
              </span>
            </div>
            <div className="grid grid-cols-8 gap-1">
              {yearWeeks.map((week) => (
                <WeekCell
                  key={week.week_id}
                  week={week}
                  isStart={week.week_id === startWeek}
                  isEnd={week.week_id === endWeek}
                  isInRange={isInRange(week.week_id)}
                  onClick={() => handleClick(week.week_id, week.status)}
                />
              ))}
            </div>
          </div>
        )
      })}

      {/* 圖例 */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-100 border border-green-300" />
          <span>Available</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-100 border border-yellow-300" />
          <span>Fallback</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-gray-200" />
          <span>Not allowed</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-blue-500" />
          <span>Selected</span>
        </div>
      </div>
    </div>
  )
}

interface WeekCellProps {
  week: WeekStatus
  isStart: boolean
  isEnd: boolean
  isInRange: boolean
  onClick: () => void
}

function WeekCell({ week, isStart, isEnd, isInRange, onClick }: WeekCellProps) {
  const weekNum = week.week_id.slice(-2)

  const statusStyles = {
    available: 'bg-green-100 text-green-800 hover:bg-green-200 cursor-pointer',
    missing: 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200 cursor-pointer',
    not_allowed: 'bg-gray-200 text-gray-400 cursor-not-allowed',
  }

  const statusIcons = {
    available: <CheckCircle className="h-2.5 w-2.5" />,
    missing: <AlertTriangle className="h-2.5 w-2.5" />,
    not_allowed: <XCircle className="h-2.5 w-2.5" />,
  }

  return (
    <button
      onClick={onClick}
      disabled={week.status === 'not_allowed'}
      className={cn(
        'p-1.5 rounded text-xs text-center transition-all relative',
        statusStyles[week.status],
        isInRange && 'ring-2 ring-blue-500 bg-blue-100 text-blue-800',
        (isStart || isEnd) && 'ring-2 ring-blue-600 bg-blue-500 text-white'
      )}
      title={getTooltip(week)}
    >
      <div className="flex items-center justify-center gap-0.5">
        {!isInRange && statusIcons[week.status]}
        <span className="font-medium">W{weekNum}</span>
      </div>
      {week.valid_ic && (
        <div className="text-[10px] opacity-80">
          {week.valid_ic.toFixed(2)}
        </div>
      )}
      {week.status === 'missing' && (
        <AlertTriangle className="absolute -top-1 -right-1 h-3 w-3 text-yellow-500" />
      )}
    </button>
  )
}

function getTooltip(week: WeekStatus): string {
  const lines = [`Week: ${week.week_id}`, `Status: ${week.status}`]

  if (week.model_name) {
    lines.push(`Model: ${week.model_name}`)
  }
  if (week.valid_ic) {
    lines.push(`Valid IC: ${week.valid_ic.toFixed(4)}`)
  }
  if (week.fallback_week) {
    lines.push(`Fallback from: ${week.fallback_week}`)
    if (week.fallback_model) {
      lines.push(`Using model: ${week.fallback_model}`)
    }
  }
  if (week.reason) {
    lines.push(`Reason: ${week.reason}`)
  }

  return lines.join('\n')
}
