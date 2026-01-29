import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

export function Factors() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">因子清單</h1>
        <button className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
          新增因子
        </button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>因子列表</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="px-4 py-3 text-left font-medium">名稱</th>
                  <th className="px-4 py-3 text-left font-medium">表達式</th>
                  <th className="px-4 py-3 text-left font-medium">IC</th>
                  <th className="px-4 py-3 text-left font-medium">狀態</th>
                  <th className="px-4 py-3 text-left font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-4 py-3">MA5</td>
                  <td className="px-4 py-3 font-mono text-xs">Mean($close, 5)</td>
                  <td className="px-4 py-3">0.035</td>
                  <td className="px-4 py-3">
                    <span className="rounded-full bg-green-100 px-2 py-1 text-xs text-green-700">
                      啟用
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button className="text-muted-foreground hover:text-foreground">
                      編輯
                    </button>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3">RSI14</td>
                  <td className="px-4 py-3 font-mono text-xs">RSI($close, 14)</td>
                  <td className="px-4 py-3">0.028</td>
                  <td className="px-4 py-3">
                    <span className="rounded-full bg-green-100 px-2 py-1 text-xs text-green-700">
                      啟用
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button className="text-muted-foreground hover:text-foreground">
                      編輯
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
