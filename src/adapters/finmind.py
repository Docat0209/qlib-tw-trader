from datetime import date
from decimal import Decimal

import httpx

from src.adapters.base import DataSourceAdapter
from src.shared.types import OHLCV


class FinMindAdapter(DataSourceAdapter):
    """FinMind 資料來源（用於補全歷史資料）"""

    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str = ""):
        self._token = token

    async def fetch_daily_ohlcv(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[OHLCV]:
        """取得日K線資料"""
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.BASE_URL, params=params, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 200:
            msg = data.get("msg", "Unknown error")
            raise RuntimeError(f"FinMind API error: {msg}")

        results = []
        for row in data.get("data", []):
            # 欄位: date, stock_id, Trading_Volume, Trading_money,
            #       open, max, min, close, spread, Trading_turnover
            ohlcv = OHLCV(
                date=date.fromisoformat(row["date"]),
                stock_id=row["stock_id"],
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["max"])),
                low=Decimal(str(row["min"])),
                close=Decimal(str(row["close"])),
                volume=int(row["Trading_Volume"]),
            )
            results.append(ohlcv)

        return results
