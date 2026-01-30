"""
系統監控 API
"""

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.system import (
    DatasetStatus,
    DataStatusResponse,
    HealthResponse,
    StockItem,
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
    session: Session = Depends(get_db),
):
    """檢查資料狀態"""
    # 所有 dataset repositories
    repos = [
        ("OHLCV", OHLCVRepository(session)),
        ("AdjClose", AdjCloseRepository(session)),
        ("PER", PERRepository(session)),
        ("Institutional", InstitutionalRepository(session)),
        ("Margin", MarginRepository(session)),
        ("Shareholding", ShareholdingRepository(session)),
    ]

    yesterday = date.today() - timedelta(days=1)

    # 1. 每個 Dataset 的整體狀態（用 global 查詢，很快）
    datasets = []
    for name, repo in repos:
        earliest = repo.get_global_earliest_date()
        latest = repo.get_global_latest_date()
        is_fresh = latest >= yesterday if latest else False
        datasets.append(
            DatasetStatus(
                name=name,
                earliest_date=earliest,
                latest_date=latest,
                is_fresh=is_fresh,
            )
        )

    # 2. 股票清單（只從 OHLCV 取，用它的最新日期判斷是否 fresh）
    ohlcv_repo = repos[0][1]
    all_stock_ids = ohlcv_repo.get_all_stock_ids()

    stocks = []
    for stock_id in all_stock_ids:
        latest = ohlcv_repo.get_latest_date(stock_id)
        is_fresh = latest >= yesterday if latest else False
        stocks.append(StockItem(stock_id=stock_id, is_fresh=is_fresh))

    return DataStatusResponse(
        datasets=datasets,
        stocks=stocks,
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
