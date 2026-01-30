"""
系統相關 Schema
"""

from datetime import date, datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """健康檢查回應"""

    status: str
    timestamp: datetime
    version: str


class DatasetStatus(BaseModel):
    """單一資料集狀態"""

    name: str
    earliest_date: date | None
    latest_date: date | None
    is_fresh: bool


class StockItem(BaseModel):
    """股票清單項目"""

    stock_id: str
    is_fresh: bool


class DataStatusResponse(BaseModel):
    """資料狀態回應"""

    datasets: list[DatasetStatus]
    stocks: list[StockItem]
    checked_at: datetime


class SyncRequest(BaseModel):
    """同步請求"""

    stock_id: str
    start_date: date
    end_date: date
    datasets: list[str] | None = None  # None = all datasets


class SyncResult(BaseModel):
    """單一資料集同步結果"""

    dataset: str
    records_fetched: int
    success: bool
    error: str | None = None


class SyncResponse(BaseModel):
    """同步回應"""

    stock_id: str
    results: list[SyncResult]
    total_records: int
    synced_at: datetime
