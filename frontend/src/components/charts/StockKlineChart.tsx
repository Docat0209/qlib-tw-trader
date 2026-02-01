import { useEffect, useRef } from 'react'
import { createChart, IChartApi, CandlestickData, SeriesMarker, Time, ISeriesApi } from 'lightweight-charts'

export interface KlinePoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface TradePoint {
  date: string
  side: 'buy' | 'sell'
  price: number
  shares: number
}

interface StockKlineChartProps {
  klines: KlinePoint[]
  trades: TradePoint[]
  height?: number
}

export function StockKlineChart({ klines, trades, height = 400 }: StockKlineChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!chartContainerRef.current) return

    // 清除舊圖表
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    // 建立圖表
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#e5e7eb' },
        horzLines: { color: '#e5e7eb' },
      },
      timeScale: {
        borderColor: '#e5e7eb',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#e5e7eb',
      },
    })

    chartRef.current = chart

    // 建立 K 線 series (lightweight-charts v4 API)
    const candlestickSeries: ISeriesApi<'Candlestick'> = chart.addCandlestickSeries({
      upColor: '#ef4444',      // 台股：上漲紅色
      downColor: '#22c55e',    // 台股：下跌綠色
      borderUpColor: '#ef4444',
      borderDownColor: '#22c55e',
      wickUpColor: '#ef4444',
      wickDownColor: '#22c55e',
    })

    // 轉換 K 線資料
    const candleData: CandlestickData[] = klines.map((k) => ({
      time: k.date as Time,
      open: k.open,
      high: k.high,
      low: k.low,
      close: k.close,
    }))

    candlestickSeries.setData(candleData)

    // 建立買賣標記
    if (trades.length > 0) {
      const markers: SeriesMarker<Time>[] = trades.map((t) => ({
        time: t.date as Time,
        position: t.side === 'buy' ? 'belowBar' : 'aboveBar',
        color: t.side === 'buy' ? '#22c55e' : '#ef4444',
        shape: t.side === 'buy' ? 'arrowUp' : 'arrowDown',
        text: `${t.side === 'buy' ? '買' : '賣'} ${t.price}`,
      }))

      candlestickSeries.setMarkers(markers)
    }

    // 自動調整時間軸
    chart.timeScale().fitContent()

    // 響應式調整
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
        chartRef.current.timeScale().fitContent()
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [klines, trades, height])

  if (klines.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground"
        style={{ height }}
      >
        Select a stock to view K-line chart
      </div>
    )
  }

  return <div ref={chartContainerRef} style={{ width: '100%', height }} />
}
