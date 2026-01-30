"""
系統監控 API
"""

from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.system import (
    DatasetStatus,
    DataStatusResponse,
    HealthResponse,
    SyncRequest,
    SyncResponse,
    SyncResult,
)
from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    OHLCVRepository,
    PERRepository,
    ShareholdingRepository,
)
from src.services.data_service import DataService, Dataset

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


# Dataset name to enum mapping
DATASET_MAP = {
    "ohlcv": Dataset.OHLCV,
    "adj_close": Dataset.ADJ_CLOSE,
    "per": Dataset.PER,
    "institutional": Dataset.INSTITUTIONAL,
    "margin": Dataset.MARGIN,
    "shareholding": Dataset.SHAREHOLDING,
}


@router.post("/sync", response_model=SyncResponse)
async def sync_data(
    request: SyncRequest,
    session: Session = Depends(get_db),
):
    """同步指定股票的資料"""
    service = DataService(session)

    # 決定要同步的資料集
    if request.datasets:
        datasets_to_sync = [
            (name, DATASET_MAP[name.lower()])
            for name in request.datasets
            if name.lower() in DATASET_MAP
        ]
    else:
        datasets_to_sync = list(DATASET_MAP.items())

    results = []
    total_records = 0

    for name, dataset in datasets_to_sync:
        try:
            data = await service.get(
                dataset=dataset,
                stock_id=request.stock_id,
                start_date=request.start_date,
                end_date=request.end_date,
                auto_fetch=True,
            )
            count = len(data)
            total_records += count
            results.append(
                SyncResult(
                    dataset=name,
                    records_fetched=count,
                    success=True,
                )
            )
        except Exception as e:
            results.append(
                SyncResult(
                    dataset=name,
                    records_fetched=0,
                    success=False,
                    error=str(e),
                )
            )

    return SyncResponse(
        stock_id=request.stock_id,
        results=results,
        total_records=total_records,
        synced_at=datetime.now(),
    )
