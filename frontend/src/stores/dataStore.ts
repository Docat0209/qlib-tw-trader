import { create } from 'zustand'

type EntityType = 'factors' | 'models' | 'datasets' | 'backtests' | 'dashboard'

interface DataState {
  timestamps: Record<EntityType, number>
  invalidate: (entity: EntityType | EntityType[]) => void
}

export const useDataStore = create<DataState>((set) => ({
  timestamps: {
    factors: Date.now(),
    models: Date.now(),
    datasets: Date.now(),
    backtests: Date.now(),
    dashboard: Date.now(),
  },
  invalidate: (entity) => {
    const entities = Array.isArray(entity) ? entity : [entity]
    set((state) => {
      const newTimestamps = { ...state.timestamps }
      for (const e of entities) {
        newTimestamps[e] = Date.now()
      }
      return { timestamps: newTimestamps }
    })
  },
}))
