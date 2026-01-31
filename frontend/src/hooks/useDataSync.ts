import { useWebSocket, JobEvent } from './useWebSocket'
import { useDataStore } from '@/stores/dataStore'
import { useCallback } from 'react'

/**
 * Global data sync hook
 *
 * Listens to WebSocket events and invalidates the data store
 * when data is updated. Should be called once in App.tsx.
 */
export function useDataSync() {
  const { invalidate } = useDataStore()

  const handleMessage = useCallback((event: JobEvent) => {
    // Handle data_updated events from CRUD operations
    if (event.type === 'data_updated' && event.entity) {
      invalidate(event.entity)
      // Also invalidate dashboard when models or factors change
      if (['factors', 'models'].includes(event.entity)) {
        invalidate('dashboard')
      }
    }

    // Handle job completion events
    if (event.type === 'job_completed' || event.type === 'job_failed') {
      if (event.job_type === 'train') {
        invalidate(['models', 'dashboard'])
      }
      if (event.job_type === 'backtest') {
        invalidate('backtests')
      }
      if (event.job_type === 'sync') {
        invalidate(['datasets', 'dashboard'])
      }
    }
  }, [invalidate])

  useWebSocket({ onMessage: handleMessage })
}
