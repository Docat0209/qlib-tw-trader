"""
yfinance 資料來源
用於取得還原股價（Adjusted Close）
"""

from datetime import date, timedelta
from decimal import Decimal

from src.adapters.base import StockDataAdapter
from src.shared.types import AdjClose


class YFinanceAdjCloseAdapter(StockDataAdapter[AdjClose]):
    """yfinance 還原股價"""

    @property
    def source_name(self) -> str:
        return "yfinance"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPriceAdj"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[AdjClose]:
        """取得還原股價（注意：yfinance 是同步 API，這裡用 run_in_executor）"""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_sync, stock_id, start_date, end_date
        )

    def _fetch_sync(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[AdjClose]:
        """同步版本"""
        import yfinance as yf

        # yfinance 使用 .TW 後綴
        ticker = yf.Ticker(f"{stock_id}.TW")
        hist = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
        )

        if hist.empty:
            return []

        results = []
        for idx, row in hist.iterrows():
            # yfinance 返回的 Close 已經是還原價
            adj_close = row.get("Close")
            if adj_close is None:
                continue

            results.append(
                AdjClose(
                    date=idx.date(),
                    stock_id=stock_id,
                    adj_close=Decimal(str(round(adj_close, 2))),
                )
            )

        return results
