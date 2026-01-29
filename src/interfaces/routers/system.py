"""
系統監控 API
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.system import (
    DatasetStatus,
    DataStatusResponse,
    HealthResponse,
)
from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    OHLCVRepository,
    PERRepository,
    ShareholdingRepository,
)

router = APIRouter()

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(),
        version=VERSION,
    )


@router.get("/data-status", response_model=DataStatusResponse)
async def data_status(
    stock_id: str = Query("2330", description="股票代碼"),
    session: Session = Depends(get_db),
):
    """檢查指定股票的資料狀態"""
    repos = [
        ("OHLCV", OHLCVRepository(session)),
        ("AdjClose", AdjCloseRepository(session)),
        ("PER", PERRepository(session)),
        ("Institutional", InstitutionalRepository(session)),
        ("Margin", MarginRepository(session)),
        ("Shareholding", ShareholdingRepository(session)),
    ]

    datasets = []
    for name, repo in repos:
        latest = repo.get_latest_date(stock_id)
        count = repo.count(stock_id) if hasattr(repo, "count") else 0
        datasets.append(
            DatasetStatus(
                name=name,
                latest_date=latest,
                record_count=count,
            )
        )

    return DataStatusResponse(
        stock_id=stock_id,
        datasets=datasets,
        checked_at=datetime.now(),
    )
