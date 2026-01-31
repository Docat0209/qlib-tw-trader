import { useState, useEffect } from 'react'
import { X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { Factor, FactorCreate, FactorUpdate, factorApi } from '@/api/client'

interface FactorFormDialogProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: () => void
  factor: Factor | null // null = 新增, Factor = 編輯
}

const CATEGORIES = [
  { value: 'technical', label: 'Technical' },
  { value: 'chips', label: 'Chips' },
  { value: 'valuation', label: 'Valuation' },
  { value: 'revenue', label: 'Revenue' },
]

export function FactorFormDialog({ isOpen, onClose, onSubmit, factor }: FactorFormDialogProps) {
  const isEditing = factor !== null

  // Form state
  const [name, setName] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [category, setCategory] = useState('technical')
  const [expression, setExpression] = useState('')
  const [description, setDescription] = useState('')

  // Validation state
  const [validating, setValidating] = useState(false)
  const [validationResult, setValidationResult] = useState<{
    valid: boolean
    error?: string
    warnings?: string[]
  } | null>(null)

  // Submit state
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Reset form when dialog opens/closes or factor changes
  useEffect(() => {
    if (isOpen) {
      if (factor) {
        setName(factor.name)
        setDisplayName(factor.display_name || '')
        setCategory(factor.category)
        setExpression(factor.formula)
        setDescription(factor.description || '')
      } else {
        setName('')
        setDisplayName('')
        setCategory('technical')
        setExpression('')
        setDescription('')
      }
      setValidationResult(null)
      setSubmitError(null)
    }
  }, [isOpen, factor])

  // Validate expression on blur
  const handleExpressionBlur = async () => {
    if (!expression.trim()) {
      setValidationResult(null)
      return
    }

    setValidating(true)
    try {
      const result = await factorApi.validate(expression)
      setValidationResult(result)
    } catch {
      setValidationResult({ valid: false, error: 'Validation failed' })
    } finally {
      setValidating(false)
    }
  }

  // Handle form submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitError(null)

    // Validate required fields
    if (!name.trim()) {
      setSubmitError('Name is required')
      return
    }
    if (!expression.trim()) {
      setSubmitError('Expression is required')
      return
    }
    if (validationResult && !validationResult.valid) {
      setSubmitError('Please fix expression errors before submitting')
      return
    }

    setSubmitting(true)
    try {
      if (isEditing) {
        const data: FactorUpdate = {
          name: name.trim(),
          display_name: displayName.trim() || undefined,
          category,
          formula: expression.trim(),
          description: description.trim() || undefined,
        }
        const id = parseInt(factor!.id.replace('f', ''))
        await factorApi.update(id, data)
      } else {
        const data: FactorCreate = {
          name: name.trim(),
          display_name: displayName.trim() || undefined,
          category,
          formula: expression.trim(),
          description: description.trim() || undefined,
        }
        await factorApi.create(data)
      }
      onSubmit()
      onClose()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to save factor')
    } finally {
      setSubmitting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Dialog */}
      <div className="relative bg-background border border-border rounded-lg shadow-lg w-full max-w-lg mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h2 className="text-lg font-semibold">
            {isEditing ? 'Edit Factor' : 'Add Factor'}
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-secondary transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Name <span className="text-red">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., mom_5d"
              className="w-full px-3 py-2 bg-secondary border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Display Name */}
          <div>
            <label className="block text-sm font-medium mb-1">Display Name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="e.g., 5 Day Momentum"
              className="w-full px-3 py-2 bg-secondary border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Category */}
          <div>
            <label className="block text-sm font-medium mb-1">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full px-3 py-2 bg-secondary border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat.value} value={cat.value}>
                  {cat.label}
                </option>
              ))}
            </select>
          </div>

          {/* Expression */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Expression <span className="text-red">*</span>
            </label>
            <input
              type="text"
              value={expression}
              onChange={(e) => {
                setExpression(e.target.value)
                setValidationResult(null)
              }}
              onBlur={handleExpressionBlur}
              placeholder="e.g., $close / Ref($close, 5) - 1"
              className={`w-full px-3 py-2 bg-secondary border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary mono text-sm ${
                validationResult
                  ? validationResult.valid
                    ? 'border-green'
                    : 'border-red'
                  : 'border-border'
              }`}
            />
            {/* Validation feedback */}
            <div className="mt-1 min-h-[20px]">
              {validating && (
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Validating...
                </span>
              )}
              {!validating && validationResult && (
                validationResult.valid ? (
                  <span className="text-xs text-green flex items-center gap-1">
                    <CheckCircle className="h-3 w-3" />
                    Valid expression (uses: {validationResult.warnings?.length ? validationResult.warnings.join(', ') : 'ok'})
                  </span>
                ) : (
                  <span className="text-xs text-red flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {validationResult.error}
                  </span>
                )
              )}
            </div>
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={2}
              className="w-full px-3 py-2 bg-secondary border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            />
          </div>

          {/* Error message */}
          {submitError && (
            <div className="p-3 bg-red/10 border border-red/20 rounded-lg text-red text-sm">
              {submitError}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-border hover:bg-secondary transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {isEditing ? 'Save Changes' : 'Add Factor'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
