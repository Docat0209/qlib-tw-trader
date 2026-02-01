import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

export interface EquityCurvePoint {
  date: string
  equity: number
  benchmark?: number | null
  drawdown?: number | null
}

interface EquityCurveProps {
  data: EquityCurvePoint[]
  initialCapital?: number
  height?: number
  showBenchmark?: boolean
}

export function EquityCurve({
  data,
  initialCapital = 1000000,
  height = 200,
  showBenchmark = false,
}: EquityCurveProps) {
  const chartData = useMemo(() => {
    return data.map((point) => ({
      date: point.date.slice(5), // MM-DD format
      equity: point.equity,
      benchmark: point.benchmark,
      return: ((point.equity - initialCapital) / initialCapital) * 100,
    }))
  }, [data, initialCapital])

  const minEquity = useMemo(() => {
    const min = Math.min(...data.map((p) => p.equity))
    return Math.floor(min * 0.99)
  }, [data])

  const maxEquity = useMemo(() => {
    const max = Math.max(...data.map((p) => p.equity))
    return Math.ceil(max * 1.01)
  }, [data])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        No data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
        />
        <YAxis
          domain={[minEquity, maxEquity]}
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
        />
        <Tooltip
          formatter={(value: number, name: string) => {
            if (name === 'equity') {
              return [`$${value.toLocaleString()}`, 'Equity']
            }
            if (name === 'benchmark') {
              return [`$${value.toLocaleString()}`, 'Benchmark']
            }
            return [value, name]
          }}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <ReferenceLine y={initialCapital} stroke="#9ca3af" strokeDasharray="5 5" />
        <Line
          type="monotone"
          dataKey="equity"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#3b82f6' }}
        />
        {showBenchmark && (
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke="#9ca3af"
            strokeWidth={1}
            dot={false}
            strokeDasharray="5 5"
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
