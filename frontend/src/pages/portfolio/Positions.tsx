import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

export function Positions() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">當前持倉</h1>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">總持倉價值</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$1,234,567</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">持倉數量</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">5</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">今日損益</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">-$5,432</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>持倉明細</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="px-4 py-3 text-left font-medium">股票</th>
                  <th className="px-4 py-3 text-right font-medium">股數</th>
                  <th className="px-4 py-3 text-right font-medium">成本價</th>
                  <th className="px-4 py-3 text-right font-medium">現價</th>
                  <th className="px-4 py-3 text-right font-medium">損益</th>
                  <th className="px-4 py-3 text-right font-medium">報酬率</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                    暫無持倉資料
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
