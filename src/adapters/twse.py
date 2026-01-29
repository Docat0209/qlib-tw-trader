import asyncio
from datetime import date
from decimal import Decimal

import httpx

from src.adapters.base import DataSourceAdapter
from src.shared.types import OHLCV

# TWSE 限流: 每 5 秒 3 次請求
REQUEST_INTERVAL = 2.0  # 保守設定


class TwseAdapter(DataSourceAdapter):
    """台灣證券交易所資料來源"""

    BASE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    def __init__(self):
        self._last_request_time = 0.0

    async def _rate_limit(self):
        """限流控制"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            await asyncio.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _fetch_month(
        self,
        stock_id: str,
        year: int,
        month: int,
    ) -> list[OHLCV]:
        """取得單月資料"""
        await self._rate_limit()

        params = {
            "response": "json",
            "date": f"{year}{month:02d}01",
            "stockNo": stock_id,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("stat") != "OK":
            return []

        results = []
        for row in data.get("data", []):
            # 欄位: 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
            # 日期格式: 114/01/02 (民國年)
            date_parts = row[0].split("/")
            year_ad = int(date_parts[0]) + 1911
            ohlcv = OHLCV(
                date=date(year_ad, int(date_parts[1]), int(date_parts[2])),
                stock_id=stock_id,
                open=Decimal(row[3].replace(",", "")),
                high=Decimal(row[4].replace(",", "")),
                low=Decimal(row[5].replace(",", "")),
                close=Decimal(row[6].replace(",", "")),
                volume=int(row[1].replace(",", "")),
            )
            results.append(ohlcv)

        return results

    async def fetch_daily_ohlcv(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[OHLCV]:
        """取得日K線資料（跨月自動處理）"""
        results = []
        current = date(start_date.year, start_date.month, 1)

        while current <= end_date:
            month_data = await self._fetch_month(
                stock_id, current.year, current.month
            )
            for ohlcv in month_data:
                if start_date <= ohlcv.date <= end_date:
                    results.append(ohlcv)

            # 下個月
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return results
