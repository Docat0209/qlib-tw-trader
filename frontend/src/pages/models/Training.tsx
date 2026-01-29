import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

export function Training() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">訓練記錄</h1>
        <button className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
          開始訓練
        </button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>訓練歷史</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="px-4 py-3 text-left font-medium">ID</th>
                  <th className="px-4 py-3 text-left font-medium">開始時間</th>
                  <th className="px-4 py-3 text-left font-medium">完成時間</th>
                  <th className="px-4 py-3 text-left font-medium">模型 IC</th>
                  <th className="px-4 py-3 text-left font-medium">選中因子</th>
                  <th className="px-4 py-3 text-left font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                    暫無訓練記錄
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
