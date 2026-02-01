from datetime import date

from pydantic import BaseModel


class TradeItem(BaseModel):
    """交易項目"""

    id: int
    date: date
    symbol: str
    name: str | None
    side: str
    shares: int
    price: float
    amount: float
    commission: float
    reason: str | None


class TradesResponse(BaseModel):
    """交易紀錄回應"""

    items: list[TradeItem]
    total: int
