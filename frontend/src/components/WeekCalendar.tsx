import { cn } from '@/lib/utils'
import { WeekSlot } from '@/api/client'
import { CheckCircle, AlertTriangle, Clock, PlayCircle, XCircle, Loader2 } from 'lucide-react'

interface WeekCalendarProps {
  slots: WeekSlot[]
  selected: string | null
  onSelect: (weekId: string) => void
  currentFactorPoolHash: string | null
  onTrainYear?: (year: string) => void
  isTraining?: boolean
  queueYear?: string | null
  queueRemaining?: number
  onCancelQueue?: () => void
}

export function WeekCalendar({ slots, selected, onSelect, currentFactorPoolHash, onTrainYear, isTraining, queueYear, queueRemaining, onCancelQueue }: WeekCalendarProps) {
  // 按年份分組
  const byYear = slots.reduce((acc, slot) => {
    const year = slot.week_id.slice(0, 4)
    if (!acc[year]) acc[year] = []
    acc[year].push(slot)
    return acc
  }, {} as Record<string, WeekSlot[]>)

  // 每年內按週數降序排列（W52, W51, ...）
  Object.values(byYear).forEach(yearSlots => {
    yearSlots.sort((a, b) => b.week_id.localeCompare(a.week_id))
  })

  // 按年份降序排列
  const sortedYears = Object.keys(byYear).sort((a, b) => b.localeCompare(a))

  return (
    <div className="space-y-4">
      {sortedYears.map((year) => {
        const yearSlots = byYear[year]
        const trainedCount = yearSlots.filter(s => s.status === 'trained').length
        const trainableCount = yearSlots.filter(s => s.status === 'trainable').length
        const totalCount = yearSlots.length

        return (
          <div key={year}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-sm text-muted-foreground">{year}</h3>
                <span className="text-[10px] text-muted-foreground">
                  <span className="text-green-600">{trainedCount}</span>
                  {trainableCount > 0 && <span className="text-gray-500">+{trainableCount}</span>}
                  /{totalCount}
                </span>
              </div>
              {queueYear === year ? (
                // 正在批次訓練此年
                <div className="flex items-center gap-2">
                  <span className="flex items-center gap-1 text-xs text-blue-600">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {queueRemaining !== undefined ? queueRemaining + 1 : '?'} left
                  </span>
                  {onCancelQueue && (
                    <button
                      onClick={onCancelQueue}
                      className="flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-red-50 text-red-600 hover:bg-red-100"
                      title="Cancel remaining training"
                    >
                      <XCircle className="h-3 w-3" />
                    </button>
                  )}
                </div>
              ) : onTrainYear && trainableCount > 0 && (
                <button
                  onClick={() => onTrainYear(year)}
                  disabled={isTraining || !!queueYear}
                  className={cn(
                    "flex items-center gap-1 text-xs px-2 py-0.5 rounded",
                    "bg-blue-50 text-blue-600 hover:bg-blue-100",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                  title={`Train all ${trainableCount} pending weeks of ${year}`}
                >
                  <PlayCircle className="h-3 w-3" />
                  Train {trainableCount}
                </button>
              )}
            </div>
            <div className="grid grid-cols-8 gap-1">
              {yearSlots.map((slot) => (
                <WeekCell
                  key={slot.week_id}
                  slot={slot}
                  isSelected={slot.week_id === selected}
                  onClick={() => onSelect(slot.week_id)}
                  currentFactorPoolHash={currentFactorPoolHash}
                />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

interface WeekCellProps {
  slot: WeekSlot
  isSelected: boolean
  onClick: () => void
  currentFactorPoolHash: string | null
}

function WeekCell({ slot, isSelected, onClick, currentFactorPoolHash }: WeekCellProps) {
  const weekNum = slot.week_id.slice(-2)
  const isOutdated = slot.model?.is_outdated ||
    (slot.model && currentFactorPoolHash && slot.model.factor_pool_hash !== currentFactorPoolHash)

  const statusStyles = {
    trained: 'bg-green-100 text-green-800 hover:bg-green-200',
    trainable: 'bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200',
    insufficient_data: 'bg-gray-200 text-gray-400 cursor-not-allowed',
  }

  const statusIcons = {
    trained: <CheckCircle className="h-2.5 w-2.5" />,
    trainable: <Clock className="h-2.5 w-2.5" />,
    insufficient_data: null,
  }

  return (
    <button
      onClick={onClick}
      disabled={slot.status === 'insufficient_data'}
      className={cn(
        'p-1.5 rounded text-xs text-center transition-all relative',
        statusStyles[slot.status],
        isSelected && 'ring-2 ring-blue-500 ring-offset-1',
        isOutdated && 'border-2 border-yellow-400'
      )}
      title={getTooltip(slot, isOutdated)}
    >
      <div className="flex items-center justify-center gap-0.5">
        {statusIcons[slot.status]}
        <span className="font-medium">W{weekNum}</span>
      </div>
      {slot.model && (
        <div className="text-[10px] opacity-80">
          {slot.model.model_ic.toFixed(2)}
        </div>
      )}
      {isOutdated && (
        <AlertTriangle className="absolute -top-1 -right-1 h-3 w-3 text-yellow-500" />
      )}
    </button>
  )
}

function getTooltip(slot: WeekSlot, isOutdated: boolean | undefined): string {
  const lines = [
    `Week: ${slot.week_id}`,
    `Valid: ${slot.valid_start} ~ ${slot.valid_end}`,
    `Train: ${slot.train_start} ~ ${slot.train_end}`,
    `Status: ${slot.status}`,
  ]

  if (slot.model) {
    lines.push(`IC: ${slot.model.model_ic.toFixed(4)}`)
    lines.push(`Factors: ${slot.model.factor_count}`)
    if (isOutdated) {
      lines.push('Warning: Factor pool changed')
    }
  }

  return lines.join('\n')
}
