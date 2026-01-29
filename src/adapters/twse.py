"""
TWSE 資料來源
- Bulk adapters: 一次取全市場當日資料（OpenAPI）
- Stock adapter: 取單一股票歷史資料（舊版 API）
"""

import asyncio
from datetime import date
from decimal import Decimal, InvalidOperation

import httpx

from src.adapters.base import BulkDataAdapter, StockDataAdapter
from src.shared.types import (
    Institutional,
    Margin,
    OHLCV,
    PER,
    Shareholding,
)

# TWSE 限流: 每 5 秒 3 次請求
REQUEST_INTERVAL = 2.0


def parse_roc_date(raw: str) -> date | None:
    """解析民國年日期（7位: 1150129 或 斜線: 115/01/29）"""
    if not raw:
        return None
    try:
        if "/" in raw:
            parts = raw.split("/")
            return date(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
        elif len(raw) == 7:
            return date(int(raw[:3]) + 1911, int(raw[3:5]), int(raw[5:7]))
    except (ValueError, IndexError):
        pass
    return None


def safe_decimal(value: str) -> Decimal | None:
    """安全轉換為 Decimal"""
    if not value or value in ("--", "-", ""):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def safe_int(value: str) -> int:
    """安全轉換為 int"""
    if not value or value in ("--", "-", ""):
        return 0
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return 0


# =============================================================================
# Bulk Adapters (RWD API - 全市場當日資料，更新較快)
# =============================================================================


async def _fetch_rwd(url: str, params: dict | None = None) -> dict | None:
    """共用的 RWD API 呼叫"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get("stat") != "OK":
            return None

        return data


class TwseBulkOHLCVAdapter(BulkDataAdapter[OHLCV]):
    """TWSE 日K線（全市場）- RWD API"""

    URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPrice"

    async def fetch_all(self, target_date: date) -> list[OHLCV]:
        """取得當日全市場日K"""
        data = await _fetch_rwd(self.URL)
        if not data:
            return []

        # 日期在 data.date 欄位（格式: 20260129）
        data_date_str = data.get("date", "")
        if data_date_str:
            try:
                data_date = date(
                    int(data_date_str[:4]),
                    int(data_date_str[4:6]),
                    int(data_date_str[6:8]),
                )
                if data_date != target_date:
                    return []
            except (ValueError, IndexError):
                pass

        rows = data.get("data", [])
        results = []

        for row in rows:
            # RWD 格式: [證券代號, 證券名稱, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌(+/-), 漲跌價差, 最後揭示買價, 最後揭示買量, 最後揭示賣價, 最後揭示賣量, 本益比]
            if len(row) < 8:
                continue

            open_val = safe_decimal(row[4])
            high = safe_decimal(row[5])
            low = safe_decimal(row[6])
            close = safe_decimal(row[7])
            volume = safe_int(row[2])

            if open_val is None or close is None:
                continue

            results.append(
                OHLCV(
                    date=target_date,
                    stock_id=row[0].strip(),
                    open=open_val,
                    high=high or open_val,
                    low=low or open_val,
                    close=close,
                    volume=volume,
                )
            )

        return results


class TwseBulkPERAdapter(BulkDataAdapter[PER]):
    """TWSE PER/PBR/殖利率（全市場）- RWD API"""

    URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_ALL"

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPER"

    async def fetch_all(self, target_date: date) -> list[PER]:
        data = await _fetch_rwd(self.URL)
        if not data:
            return []

        # 日期在 data.date 欄位
        data_date_str = data.get("date", "")
        if data_date_str:
            try:
                data_date = date(
                    int(data_date_str[:4]),
                    int(data_date_str[4:6]),
                    int(data_date_str[6:8]),
                )
                if data_date != target_date:
                    return []
            except (ValueError, IndexError):
                pass

        rows = data.get("data", [])
        results = []

        for row in rows:
            # RWD 格式: [證券代號, 證券名稱, 本益比, 殖利率(%), 股價淨值比]
            # 範例: ['2330', '台積電', '29.50', '0.94', '9.36']
            if len(row) < 5:
                continue

            results.append(
                PER(
                    date=target_date,
                    stock_id=row[0].strip(),
                    pe_ratio=safe_decimal(row[2]),
                    dividend_yield=safe_decimal(row[3]),
                    pb_ratio=safe_decimal(row[4]),
                )
            )

        return results


class TwseBulkInstitutionalAdapter(BulkDataAdapter[Institutional]):
    """TWSE 三大法人買賣超（全市場）- RWD API"""

    URL = "https://www.twse.com.tw/rwd/zh/fund/TWT43U"

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockInstitutionalInvestorsBuySell"

    async def fetch_all(self, target_date: date) -> list[Institutional]:
        # RWD 需要帶日期參數
        date_str = f"{target_date.year}{target_date.month:02d}{target_date.day:02d}"
        data = await _fetch_rwd(self.URL, {"date": date_str, "selectType": "ALLBUT0999"})
        if not data:
            return []

        rows = data.get("data", [])
        results = []

        for row in rows:
            # RWD 格式: [證券代號, 證券名稱, 外資買進, 外資賣出, 外資淨買, 投信買進, 投信賣出, 投信淨買, 自營商買進, 自營商賣出, 自營商淨買, 三大法人淨買]
            if len(row) < 11:
                continue

            results.append(
                Institutional(
                    date=target_date,
                    stock_id=row[0].strip(),
                    foreign_buy=safe_int(row[2]),
                    foreign_sell=safe_int(row[3]),
                    trust_buy=safe_int(row[5]),
                    trust_sell=safe_int(row[6]),
                    dealer_buy=safe_int(row[8]),
                    dealer_sell=safe_int(row[9]),
                )
            )

        return results


class TwseBulkMarginAdapter(BulkDataAdapter[Margin]):
    """TWSE 融資融券（全市場）- RWD API"""

    URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockMarginPurchaseShortSale"

    async def fetch_all(self, target_date: date) -> list[Margin]:
        date_str = f"{target_date.year}{target_date.month:02d}{target_date.day:02d}"
        data = await _fetch_rwd(self.URL, {"date": date_str, "selectType": "ALL"})
        if not data:
            return []

        # MI_MARGN 有多個表格，取第二個（個股明細）
        tables = data.get("tables", [])
        if len(tables) < 2:
            return []

        rows = tables[1].get("data", [])
        results = []

        for row in rows:
            # 格式: [股票代號, 股票名稱, 融資買進, 融資賣出, 融資現金償還, 融資前日餘額, 融資今日餘額, 融資金額,
            #        融券賣出, 融券買進, 融券現券償還, 融券前日餘額, 融券今日餘額, 資券相抵, ...]
            # 範例: ['0050', '元大台灣50', '423', '375', '4', '6,946', '6,990', '3,965,750', '2', '59', '0', '185', '242', ...]
            if len(row) < 13:
                continue

            results.append(
                Margin(
                    date=target_date,
                    stock_id=row[0].strip(),
                    margin_buy=safe_int(row[2]),
                    margin_sell=safe_int(row[3]),
                    margin_balance=safe_int(row[6]),
                    short_sell=safe_int(row[8]),  # 融券賣出
                    short_buy=safe_int(row[9]),   # 融券買進
                    short_balance=safe_int(row[12]),
                )
            )

        return results


class TwseBulkShareholdingAdapter(BulkDataAdapter[Shareholding]):
    """TWSE 外資持股（全市場）- RWD API"""

    URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockShareholding"

    async def fetch_all(self, target_date: date) -> list[Shareholding]:
        date_str = f"{target_date.year}{target_date.month:02d}{target_date.day:02d}"
        data = await _fetch_rwd(self.URL, {"date": date_str, "selectType": "ALLBUT0999"})
        if not data:
            return []

        rows = data.get("data", [])
        results = []

        for row in rows:
            # 格式: [證券代號, 證券名稱, ISIN Code, 發行股數, 外資及陸資持股數, 全體外資及陸資持股異動數, 外資持股比率(%), ...]
            # 範例: ['0050', '元大台灣50', 'TW0000050004', '15,863,000,000', '15,390,965,969', '472,034,031', 97.02, ...]
            if len(row) < 7:
                continue

            # 持股比率可能是數字或字串
            ratio_val = row[6]
            if isinstance(ratio_val, (int, float)):
                ratio = Decimal(str(ratio_val))
            else:
                ratio = safe_decimal(ratio_val)

            results.append(
                Shareholding(
                    date=target_date,
                    stock_id=row[0].strip(),
                    foreign_shares=safe_int(row[4]),
                    foreign_ratio=ratio or Decimal("0"),
                )
            )

        return results


# =============================================================================
# Stock Adapter (舊版 API - 單一股票歷史資料)
# =============================================================================


class TwseStockOHLCVAdapter(StockDataAdapter[OHLCV]):
    """TWSE 日K線（單一股票歷史）"""

    BASE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    def __init__(self):
        self._last_request_time = 0.0

    @property
    def source_name(self) -> str:
        return "twse"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPrice"

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
            resp = await client.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

        if data.get("stat") != "OK":
            return []

        results = []
        for row in data.get("data", []):
            # 欄位: 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
            row_date = parse_roc_date(row[0])
            if row_date is None:
                continue

            open_val = safe_decimal(row[3])
            high = safe_decimal(row[4])
            low = safe_decimal(row[5])
            close = safe_decimal(row[6])
            volume = safe_int(row[1])

            if open_val is None or close is None:
                continue

            results.append(
                OHLCV(
                    date=row_date,
                    stock_id=stock_id,
                    open=open_val,
                    high=high or open_val,
                    low=low or open_val,
                    close=close,
                    volume=volume,
                )
            )

        return results

    async def fetch(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[OHLCV]:
        """取得日K線資料（跨月自動處理）"""
        results = []
        current = date(start_date.year, start_date.month, 1)

        while current <= end_date:
            month_data = await self._fetch_month(stock_id, current.year, current.month)
            for ohlcv in month_data:
                if start_date <= ohlcv.date <= end_date:
                    results.append(ohlcv)

            # 下個月
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return results
