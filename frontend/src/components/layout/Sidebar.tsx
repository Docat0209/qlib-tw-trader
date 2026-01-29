import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Brain,
  LineChart,
  Briefcase,
  TrendingUp,
  Database,
  Settings,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  {
    name: '模型管理',
    items: [
      { name: '因子清單', href: '/models/factors', icon: Brain },
      { name: '訓練記錄', href: '/models/training', icon: LineChart },
    ],
  },
  {
    name: '持倉控管',
    items: [
      { name: '當前持倉', href: '/portfolio/positions', icon: Briefcase },
      { name: '收益分析', href: '/portfolio/performance', icon: TrendingUp },
    ],
  },
  {
    name: '系統監控',
    items: [
      { name: '資料狀態', href: '/system/data-status', icon: Database },
    ],
  },
]

export function Sidebar() {
  return (
    <div className="flex w-64 flex-col border-r bg-card">
      <div className="flex h-16 items-center border-b px-6">
        <h1 className="text-xl font-bold">qlib-tw-trader</h1>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => (
          'href' in item ? (
            <NavItem key={item.name} item={item} />
          ) : (
            <div key={item.name} className="pt-4">
              <p className="px-3 text-xs font-semibold uppercase text-muted-foreground">
                {item.name}
              </p>
              <div className="mt-2 space-y-1">
                {item.items.map((subItem) => (
                  <NavItem key={subItem.name} item={subItem} />
                ))}
              </div>
            </div>
          )
        ))}
      </nav>
      <div className="border-t p-3">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
              isActive
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            )
          }
        >
          <Settings className="h-4 w-4" />
          設定
        </NavLink>
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
        cn(
          'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
          isActive
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
        )
      }
    >
      <Icon className="h-4 w-4" />
      {item.name}
    </NavLink>
  )
}
