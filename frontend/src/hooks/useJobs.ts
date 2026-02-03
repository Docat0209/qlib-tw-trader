import { useState, useCallback, useEffect } from 'react'
import { useWebSocket, JobEvent } from './useWebSocket'

export interface Job {
  id: string
  job_type: string
  status: string
  progress: number
  message?: string
  result?: unknown
  error?: string
}

interface JobApiResponse {
  id: string
  job_type: string
  status: string
  progress: number
  message?: string
  result?: string
}

export function useJobs() {
  const [jobs, setJobs] = useState<Map<string, Job>>(new Map())
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  // 初始化時從 API 獲取運行中的任務
  useEffect(() => {
    const fetchRunningJobs = async () => {
      try {
        const res = await fetch('/api/v1/jobs')
        if (!res.ok) return
        const data = await res.json()
        const items: JobApiResponse[] = data.items || []

        // 找到運行中的任務
        const runningJob = items.find(j => j.status === 'running' || j.status === 'queued')
        if (runningJob) {
          setJobs(prev => {
            const next = new Map(prev)
            next.set(runningJob.id, {
              id: runningJob.id,
              job_type: runningJob.job_type,
              status: runningJob.status,
              progress: runningJob.progress,
              message: runningJob.message,
            })
            return next
          })
          setActiveJobId(runningJob.id)
        }
      } catch {
        // 忽略錯誤
      }
    }
    fetchRunningJobs()
  }, [])

  const handleMessage = useCallback((event: JobEvent) => {
    const jobId = event.job_id
    if (!jobId) return

    setJobs((prev) => {
      const next = new Map(prev)

      if (event.type === 'job_created') {
        next.set(jobId, {
          id: jobId,
          job_type: event.job_type || 'unknown',
          status: 'queued',
          progress: 0,
          message: event.message,
        })
        setActiveJobId(jobId)
      } else if (event.type === 'job_progress') {
        const existing = next.get(jobId)
        if (existing) {
          next.set(jobId, {
            ...existing,
            status: event.status || existing.status,
            progress: event.progress ?? existing.progress,
            message: event.message ?? existing.message,
          })
        }
      } else if (event.type === 'job_completed') {
        const existing = next.get(jobId)
        if (existing) {
          next.set(jobId, {
            ...existing,
            status: 'completed',
            progress: 100,
            result: event.result,
          })
        }
        // 不立即清除 activeJobId，讓組件有機會處理完成狀態
        // 組件應調用 clearJob 來清除
      } else if (event.type === 'job_failed') {
        const existing = next.get(jobId)
        if (existing) {
          next.set(jobId, {
            ...existing,
            status: 'failed',
            error: event.error,
          })
        }
        // 不立即清除 activeJobId，讓組件有機會處理失敗狀態
        // 組件應調用 clearJob 來清除
      } else if (event.type === 'job_cancelled') {
        // 取消的任務直接從列表移除
        next.delete(jobId)
        if (activeJobId === jobId) {
          setActiveJobId(null)
        }
      }

      return next
    })
  }, [activeJobId])

  const { isConnected } = useWebSocket({
    onMessage: handleMessage,
  })

  const getJob = useCallback((jobId: string) => jobs.get(jobId), [jobs])

  const getActiveJob = useCallback(() => {
    if (!activeJobId) return null
    return jobs.get(activeJobId) || null
  }, [activeJobId, jobs])

  const clearJob = useCallback((jobId: string) => {
    setJobs((prev) => {
      const next = new Map(prev)
      next.delete(jobId)
      return next
    })
    if (activeJobId === jobId) {
      setActiveJobId(null)
    }
  }, [activeJobId])

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      const res = await fetch(`/api/v1/jobs/${jobId}`, { method: 'DELETE' })
      if (res.ok) {
        // WebSocket 會收到 job_cancelled 事件並清除
        // 但為了即時反應，先手動清除
        clearJob(jobId)
        return true
      }
      return false
    } catch {
      return false
    }
  }, [clearJob])

  return {
    jobs: Array.from(jobs.values()),
    activeJob: getActiveJob(),
    getJob,
    clearJob,
    cancelJob,
    isConnected,
  }
}
