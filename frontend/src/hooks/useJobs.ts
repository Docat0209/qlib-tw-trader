import { useState, useCallback } from 'react'
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

export function useJobs() {
  const [jobs, setJobs] = useState<Map<string, Job>>(new Map())
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

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
        if (activeJobId === jobId) {
          setActiveJobId(null)
        }
      } else if (event.type === 'job_failed') {
        const existing = next.get(jobId)
        if (existing) {
          next.set(jobId, {
            ...existing,
            status: 'failed',
            error: event.error,
          })
        }
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

  return {
    jobs: Array.from(jobs.values()),
    activeJob: getActiveJob(),
    getJob,
    clearJob,
    isConnected,
  }
}
