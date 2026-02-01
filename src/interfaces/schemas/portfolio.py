from __future__ import annotations

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


# Prediction Types


class PredictionRequest(BaseModel):
    """預測請求"""

    model_id: int
    top_k: int = 10
    trade_date: date | None = None  # 預計交易日期，None = 使用最新資料日期的下一天


class PredictionSignal(BaseModel):
    """預測訊號"""

    rank: int
    symbol: str
    name: str | None
    score: float


class PredictionsResponse(BaseModel):
    """預測回應"""

    trade_date: str  # 預計交易日期
    feature_date: str  # 實際使用的特徵資料日期
    model_name: str
    signals: list[PredictionSignal]
