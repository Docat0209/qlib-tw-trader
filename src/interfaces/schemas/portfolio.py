from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel


class PositionItem(BaseModel):
    """持倉項目"""

    symbol: str
    name: str | None
    shares: int
    avg_cost: float
    current_price: float | None
    market_value: float | None
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    weight: float | None


class PositionsResponse(BaseModel):
    """持倉回應"""

    as_of: date
    total_value: float
    cash: float
    positions: list[PositionItem]


class PredictionSignal(BaseModel):
    """預測信號"""

    symbol: str
    name: str | None
    score: float
    rank: int
    signal: str
    current_position: int | None


class PredictionsLatestResponse(BaseModel):
    """最新預測回應"""

    date: date
    model_id: str
    signals: list[PredictionSignal]


class PredictionHistoryItem(BaseModel):
    """歷史預測項目"""

    date: date
    symbol: str
    score: float
    rank: int
    signal: str


class PredictionsHistoryResponse(BaseModel):
    """歷史預測回應"""

    items: list[PredictionHistoryItem]
    total: int


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
