import { useEffect, useRef, useState, useCallback } from 'react'

export interface JobEvent {
  type: 'job_created' | 'job_progress' | 'job_completed' | 'job_failed' | 'job_cancelled' | 'pong'
  job_id?: string
  job_type?: string
  status?: string
  progress?: number
  message?: string
  result?: unknown
  error?: string
}

interface UseWebSocketOptions {
  onMessage?: (event: JobEvent) => void
  onConnect?: () => void
  onDisconnect?: () => void
  autoReconnect?: boolean
  reconnectInterval?: number
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<JobEvent | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/api/v1/ws`)

    ws.onopen = () => {
      setIsConnected(true)
      onConnect?.()
    }

    ws.onclose = () => {
      setIsConnected(false)
      onDisconnect?.()

      if (autoReconnect) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }

    ws.onerror = () => {
      ws.close()
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as JobEvent
        setLastEvent(data)
        onMessage?.(data)
      } catch {
        // Ignore parse errors
      }
    }

    wsRef.current = ws
  }, [autoReconnect, reconnectInterval, onConnect, onDisconnect, onMessage])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const ping = useCallback(() => {
    send({ type: 'ping' })
  }, [send])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    isConnected,
    lastEvent,
    send,
    ping,
    connect,
    disconnect,
  }
}
