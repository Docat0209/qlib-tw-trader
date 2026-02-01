import { useEffect, useRef } from 'react'
import { useDataStore } from '@/stores/dataStore'

type EntityType = 'factors' | 'models' | 'datasets' | 'backtests' | 'dashboard' | 'hyperparams'

/**
 * Auto-fetch hook that triggers fetchFn when data is invalidated
 *
 * @param entity - The entity type to watch
 * @param fetchFn - The fetch function to call when data changes
 */
export function useFetchOnChange(entity: EntityType, fetchFn: () => void) {
  const timestamp = useDataStore((state) => state.timestamps[entity])
  const isFirstRun = useRef(true)

  useEffect(() => {
    // Skip the first run (initial mount)
    if (isFirstRun.current) {
      isFirstRun.current = false
      return
    }
    fetchFn()
  }, [timestamp, fetchFn])
}
