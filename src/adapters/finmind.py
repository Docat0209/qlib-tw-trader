"""
FinMind 資料來源
用於補全歷史資料（TWSE 只提供當日 bulk 資料）
"""

from datetime import date
from decimal import Decimal

import httpx

from src.adapters.base import StockDataAdapter
from src.shared.types import (
    Institutional,
    Margin,
    MonthlyRevenue,
    OHLCV,
    PER,
    SecuritiesLending,
    Shareholding,
)

BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def safe_decimal(value) -> Decimal | None:
    """安全轉換為 Decimal"""
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def safe_int(value) -> int:
    """安全轉換為 int"""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except Exception:
        return 0


class FinMindBaseAdapter:
    """FinMind 共用功能"""

    def __init__(self, token: str = ""):
        self._token = token

    async def _fetch(self, dataset: str, params: dict) -> list[dict]:
        """共用的 API 呼叫"""
        request_params = {"dataset": dataset, **params}
        if self._token:
            request_params["token"] = self._token

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                BASE_URL, params=request_params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 200:
            msg = data.get("msg", "Unknown error")
            raise RuntimeError(f"FinMind API error: {msg}")

        return data.get("data", [])


# =============================================================================
# Stock Data Adapters (個股)
# =============================================================================


class FinMindOHLCVAdapter(FinMindBaseAdapter, StockDataAdapter[OHLCV]):
    """FinMind 日K線"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPrice"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[OHLCV]:
        rows = await self._fetch(
            "TaiwanStockPrice",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        results = []
        for row in rows:
            open_val = safe_decimal(row.get("open"))
            close = safe_decimal(row.get("close"))
            if open_val is None or close is None:
                continue

            results.append(
                OHLCV(
                    date=date.fromisoformat(row["date"]),
                    stock_id=row["stock_id"],
                    open=open_val,
                    high=safe_decimal(row.get("max")) or open_val,
                    low=safe_decimal(row.get("min")) or open_val,
                    close=close,
                    volume=safe_int(row.get("Trading_Volume")),
                )
            )

        return results


class FinMindPERAdapter(FinMindBaseAdapter, StockDataAdapter[PER]):
    """FinMind PER/PBR/殖利率"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockPER"

    async def fetch(self, stock_id: str, start_date: date, end_date: date) -> list[PER]:
        rows = await self._fetch(
            "TaiwanStockPER",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        results = []
        for row in rows:
            results.append(
                PER(
                    date=date.fromisoformat(row["date"]),
                    stock_id=row["stock_id"],
                    pe_ratio=safe_decimal(row.get("PER")),
                    pb_ratio=safe_decimal(row.get("PBR")),
                    dividend_yield=safe_decimal(row.get("dividend_yield")),
                )
            )

        return results


class FinMindInstitutionalAdapter(FinMindBaseAdapter, StockDataAdapter[Institutional]):
    """FinMind 三大法人"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockInstitutionalInvestorsBuySell"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[Institutional]:
        rows = await self._fetch(
            "TaiwanStockInstitutionalInvestorsBuySell",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        # FinMind 每種法人一列，需要聚合
        # name: Foreign_Investor, Investment_Trust, Dealer_self, Dealer_Hedging, Foreign_Dealer_Self
        grouped: dict[tuple, dict] = {}
        for row in rows:
            key = (row["stock_id"], row["date"])
            if key not in grouped:
                grouped[key] = {
                    "stock_id": row["stock_id"],
                    "date": row["date"],
                    "foreign_buy": 0,
                    "foreign_sell": 0,
                    "trust_buy": 0,
                    "trust_sell": 0,
                    "dealer_buy": 0,
                    "dealer_sell": 0,
                }

            name = row.get("name", "")
            buy = safe_int(row.get("buy"))
            sell = safe_int(row.get("sell"))

            if name == "Foreign_Investor":
                grouped[key]["foreign_buy"] += buy
                grouped[key]["foreign_sell"] += sell
            elif name == "Investment_Trust":
                grouped[key]["trust_buy"] += buy
                grouped[key]["trust_sell"] += sell
            elif name in ("Dealer_self", "Dealer_Hedging"):
                grouped[key]["dealer_buy"] += buy
                grouped[key]["dealer_sell"] += sell

        results = []
        for data in grouped.values():
            results.append(
                Institutional(
                    date=date.fromisoformat(data["date"]),
                    stock_id=data["stock_id"],
                    foreign_buy=data["foreign_buy"],
                    foreign_sell=data["foreign_sell"],
                    trust_buy=data["trust_buy"],
                    trust_sell=data["trust_sell"],
                    dealer_buy=data["dealer_buy"],
                    dealer_sell=data["dealer_sell"],
                )
            )

        # 按日期排序
        results.sort(key=lambda x: x.date)
        return results


class FinMindMarginAdapter(FinMindBaseAdapter, StockDataAdapter[Margin]):
    """FinMind 融資融券"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockMarginPurchaseShortSale"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[Margin]:
        rows = await self._fetch(
            "TaiwanStockMarginPurchaseShortSale",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        results = []
        for row in rows:
            results.append(
                Margin(
                    date=date.fromisoformat(row["date"]),
                    stock_id=row["stock_id"],
                    margin_buy=safe_int(row.get("MarginPurchaseBuy")),
                    margin_sell=safe_int(row.get("MarginPurchaseSell")),
                    margin_balance=safe_int(row.get("MarginPurchaseTodayBalance")),
                    short_buy=safe_int(row.get("ShortSaleBuy")),
                    short_sell=safe_int(row.get("ShortSaleSell")),
                    short_balance=safe_int(row.get("ShortSaleTodayBalance")),
                )
            )

        return results


class FinMindShareholdingAdapter(FinMindBaseAdapter, StockDataAdapter[Shareholding]):
    """FinMind 外資持股"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockShareholding"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[Shareholding]:
        rows = await self._fetch(
            "TaiwanStockShareholding",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        results = []
        for row in rows:
            ratio = safe_decimal(row.get("ForeignInvestmentSharesRatio"))
            results.append(
                Shareholding(
                    date=date.fromisoformat(row["date"]),
                    stock_id=row["stock_id"],
                    total_shares=safe_int(row.get("NumberOfSharesIssued")),
                    foreign_shares=safe_int(row.get("ForeignInvestmentShares")),
                    foreign_ratio=ratio or Decimal("0"),
                    foreign_remaining_shares=safe_int(row.get("ForeignInvestmentRemainingShares")),
                    foreign_remaining_ratio=safe_decimal(row.get("ForeignInvestmentRemainRatio")) or Decimal("0"),
                    foreign_upper_limit_ratio=safe_decimal(row.get("ForeignInvestmentUpperLimitRatio")) or Decimal("0"),
                    chinese_upper_limit_ratio=safe_decimal(row.get("ChineseInvestmentUpperLimitRatio")) or Decimal("0"),
                )
            )

        return results


class FinMindSecuritiesLendingAdapter(FinMindBaseAdapter, StockDataAdapter[SecuritiesLending]):
    """FinMind 借券明細（每日聚合）"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockSecuritiesLending"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[SecuritiesLending]:
        rows = await self._fetch(
            "TaiwanStockSecuritiesLending",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        # FinMind 返回每筆借券成交，需按日期聚合
        grouped: dict[tuple, int] = {}
        for row in rows:
            key = (row["stock_id"], row["date"])
            volume = safe_int(row.get("volume"))  # 小寫 volume
            grouped[key] = grouped.get(key, 0) + volume

        results = []
        for (stock_id, row_date), total_volume in grouped.items():
            results.append(
                SecuritiesLending(
                    date=date.fromisoformat(row_date),
                    stock_id=stock_id,
                    lending_volume=total_volume,
                )
            )

        results.sort(key=lambda x: x.date)
        return results


class FinMindMonthlyRevenueAdapter(FinMindBaseAdapter, StockDataAdapter[MonthlyRevenue]):
    """FinMind 月營收"""

    @property
    def source_name(self) -> str:
        return "finmind"

    @property
    def dataset_name(self) -> str:
        return "TaiwanStockMonthRevenue"

    async def fetch(
        self, stock_id: str, start_date: date, end_date: date
    ) -> list[MonthlyRevenue]:
        rows = await self._fetch(
            "TaiwanStockMonthRevenue",
            {
                "data_id": stock_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        results = []
        for row in rows:
            revenue = safe_decimal(row.get("revenue"))
            if revenue is None:
                continue

            # 使用 revenue_year 和 revenue_month（營收所屬月份）
            # 而非 date（公布日期）
            results.append(
                MonthlyRevenue(
                    stock_id=row["stock_id"],
                    year=safe_int(row.get("revenue_year")),
                    month=safe_int(row.get("revenue_month")),
                    revenue=revenue,
                )
            )

        return results
