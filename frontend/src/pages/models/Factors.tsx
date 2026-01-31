import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Plus, Edit2, ToggleLeft, ToggleRight, Brain, CheckCircle, Loader2, RefreshCw } from 'lucide-react'
import { factorApi, Factor } from '@/api/client'
import { FactorFormDialog } from '@/components/factors/FactorFormDialog'

export function Factors() {
  const [factors, setFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isFormOpen, setIsFormOpen] = useState(false)
  const [editingFactor, setEditingFactor] = useState<Factor | null>(null)

  const fetchFactors = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await factorApi.list()
      setFactors(response.items)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load factors')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFactors()
  }, [])

  const handleToggle = async (factor: Factor) => {
    try {
      const id = parseInt(factor.id.replace('f', ''))
      const updated = await factorApi.toggle(id)
      setFactors(factors.map(f => f.id === factor.id ? updated : f))
    } catch (err) {
      console.error('Failed to toggle factor:', err)
    }
  }

  const handleAdd = () => {
    setEditingFactor(null)
    setIsFormOpen(true)
  }

  const handleEdit = (factor: Factor) => {
    setEditingFactor(factor)
    setIsFormOpen(true)
  }

  const handleFormSubmit = () => {
    fetchFactors()
  }

  const enabledCount = factors.filter(f => f.enabled).length
  const evaluatedFactors = factors.filter(f => f.times_evaluated > 0)
  const avgSelectionRate = evaluatedFactors.length > 0
    ? evaluatedFactors.reduce((sum, f) => sum + f.selection_rate, 0) / evaluatedFactors.length
    : 0

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red">{error}</p>
        <button className="btn btn-secondary" onClick={fetchFactors}>
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="heading text-2xl">Factor Management</h1>
          <p className="subheading mt-1">Manage your alpha factors and expressions.</p>
        </div>
        <button className="btn btn-primary" onClick={handleAdd}>
          <Plus className="h-4 w-4" />
          Add Factor
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Total Factors</span>
            <div className="icon-box icon-box-blue">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">{factors.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Enabled</span>
            <div className="icon-box icon-box-green">
              <CheckCircle className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value text-green">{enabledCount}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <span className="stat-label">Avg Selection Rate</span>
            <div className="icon-box icon-box-purple">
              <Brain className="h-4 w-4" />
            </div>
          </div>
          <p className="stat-value">
            {evaluatedFactors.length > 0 ? `${(avgSelectionRate * 100).toFixed(0)}%` : '—'}
          </p>
          {evaluatedFactors.length > 0 && (
            <p className="text-xs text-muted-foreground">{evaluatedFactors.length} evaluated</p>
          )}
        </div>
      </div>

      {/* Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Factor List</CardTitle>
          <span className="badge badge-blue">{factors.length} factors</span>
        </CardHeader>
        <CardContent className="p-0">
          {factors.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Brain className="h-12 w-12 mb-4 opacity-50" />
              <p>No factors defined yet</p>
              <p className="text-sm">Click "Add Factor" to create your first factor.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Name</th>
                    <th className="table-header px-5 py-3 text-left">Category</th>
                    <th className="table-header px-5 py-3 text-left">Formula</th>
                    <th className="table-header px-5 py-3 text-right">Selection Rate</th>
                    <th className="table-header px-5 py-3 text-center">Status</th>
                    <th className="table-header px-5 py-3 text-center">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {factors.map((factor) => (
                    <tr key={factor.id} className="table-row">
                      <td className="table-cell px-5">
                        <div>
                          <span className="font-semibold">{factor.name}</span>
                          {factor.display_name && (
                            <p className="text-xs text-muted-foreground">{factor.display_name}</p>
                          )}
                        </div>
                      </td>
                      <td className="table-cell px-5">
                        <span className={`badge ${
                          factor.category === 'technical' ? 'badge-blue' :
                          factor.category === 'fundamental' ? 'badge-purple' :
                          'badge-orange'
                        }`}>
                          {factor.category}
                        </span>
                      </td>
                      <td className="table-cell px-5">
                        <code className="text-xs mono px-2 py-1 rounded bg-secondary text-purple">
                          {factor.formula}
                        </code>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <div>
                          {factor.times_evaluated === 0 ? (
                            <span className="text-muted-foreground text-sm">尚未評估</span>
                          ) : (
                            <>
                              <span className={`mono font-semibold ${factor.selection_rate >= 0.5 ? 'text-green' : 'text-muted-foreground'}`}>
                                {(factor.selection_rate * 100).toFixed(0)}%
                              </span>
                              <p className="text-xs text-muted-foreground">
                                ({factor.times_selected}/{factor.times_evaluated})
                              </p>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="table-cell px-5 text-center">
                        {factor.enabled ? (
                          <span className="badge badge-green">Enabled</span>
                        ) : (
                          <span className="badge badge-gray">Disabled</span>
                        )}
                      </td>
                      <td className="table-cell px-5">
                        <div className="flex items-center justify-center gap-2">
                          <button
                            className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                            onClick={() => handleEdit(factor)}
                          >
                            <Edit2 className="h-4 w-4" />
                          </button>
                          <button
                            className="p-2 rounded-lg hover:bg-secondary transition-colors"
                            onClick={() => handleToggle(factor)}
                          >
                            {factor.enabled ? (
                              <ToggleRight className="h-4 w-4 text-green" />
                            ) : (
                              <ToggleLeft className="h-4 w-4 text-muted-foreground" />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Factor Form Dialog */}
      <FactorFormDialog
        isOpen={isFormOpen}
        onClose={() => setIsFormOpen(false)}
        onSubmit={handleFormSubmit}
        factor={editingFactor}
      />
    </div>
  )
}
