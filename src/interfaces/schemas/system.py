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
    latest_date: date | None
    record_count: int


class DataStatusResponse(BaseModel):
    """資料狀態回應"""

    stock_id: str
    datasets: list[DatasetStatus]
    checked_at: datetime
