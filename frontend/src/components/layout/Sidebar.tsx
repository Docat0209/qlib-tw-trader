import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Brain,
  LineChart,
  BarChart3,
  Briefcase,
  TrendingUp,
  Settings,
  Zap,
  Layers,
} from 'lucide-react'
import { cn } from '@/lib/utils'

type NavItemType = { name: string; href: string; icon: React.ComponentType<{ className?: string }> }
type NavGroupType = { name: string; items: NavItemType[] }
type NavigationType = NavItemType | NavGroupType

const navigation: NavigationType[] = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  {
    name: 'Models',
    items: [
      { name: 'Factors', href: '/models/factors', icon: Brain },
      { name: 'Training', href: '/models/training', icon: LineChart },
      { name: 'Backtest', href: '/models/backtest', icon: BarChart3 },
    ],
  },
  {
    name: 'Portfolio',
    items: [
      { name: 'Positions', href: '/portfolio/positions', icon: Briefcase },
      { name: 'Performance', href: '/portfolio/performance', icon: TrendingUp },
    ],
  },
  {
    name: 'System',
    items: [
      { name: 'Datasets', href: '/system/datasets', icon: Layers },
    ],
  },
]

export function Sidebar() {
  return (
    <div className="sidebar flex w-60 flex-col">
      {/* Header */}
      <div className="px-4 py-5 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="icon-box icon-box-blue">
            <Zap className="h-4 w-4" />
          </div>
          <div>
            <h1 className="text-sm font-semibold">QLIB-TW</h1>
            <p className="text-[10px] text-muted-foreground">Trader</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          if ('href' in item) {
            return <NavItem key={item.name} item={item} />
          }
          return (
            <div key={item.name} className="pt-5">
              <p className="nav-section">{item.name}</p>
              <div className="mt-1 space-y-1">
                {item.items.map((subItem) => (
                  <NavItem key={subItem.name} item={subItem} />
                ))}
              </div>
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-border">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn('nav-item', isActive && 'active')
          }
        >
          <Settings className="h-4 w-4" />
          Settings
        </NavLink>

        <div className="mt-3 px-3 py-2 rounded-lg bg-secondary">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">Version</span>
            <span className="mono text-blue">v1.0.0</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function NavItem({ item }: { item: { name: string; href: string; icon: React.ComponentType<{ className?: string }> } }) {
  const Icon = item.icon
  return (
    <NavLink
      to={item.href}
      className={({ isActive }) =>
        cn('nav-item', isActive && 'active')
      }
    >
      <Icon className="h-4 w-4" />
      <span>{item.name}</span>
    </NavLink>
  )
}
