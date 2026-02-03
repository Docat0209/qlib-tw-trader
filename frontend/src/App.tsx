import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Factors } from '@/pages/models/Factors'
import { Hyperparams } from '@/pages/models/Hyperparams'
import { Training } from '@/pages/models/Training'
import { Backtest } from '@/pages/models/Backtest'
import { WalkForwardBacktest } from '@/pages/models/WalkForwardBacktest'
import { Predictions } from '@/pages/portfolio/Predictions'
import { Datasets } from '@/pages/system/Datasets'
import { useDataSync } from '@/hooks/useDataSync'

function App() {
  useDataSync()

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="models">
          <Route path="factors" element={<Factors />} />
          <Route path="hyperparams" element={<Hyperparams />} />
          <Route path="training" element={<Training />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="walk-forward" element={<WalkForwardBacktest />} />
        </Route>
        <Route path="portfolio">
          <Route path="predictions" element={<Predictions />} />
        </Route>
        <Route path="system">
          <Route path="datasets" element={<Datasets />} />
        </Route>
      </Route>
    </Routes>
  )
}

export default App
