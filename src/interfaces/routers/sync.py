"""
資料同步 API
"""

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.repositories.models import StockDaily, StockUniverse, TradingCalendar
from src.services.sync_service import SyncService

router = APIRouter()


class SyncCalendarResponse(BaseModel):
    start_date: str
    end_date: str
    new_dates: int
    total_dates: int


class SyncStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int
    missing_dates: list[str]


class SyncBulkResponse(BaseModel):
    date: str
    total: int
    inserted: int
    error: str | None = None


class SyncAllResponse(BaseModel):
    stocks: int
    total_inserted: int
    errors: list[dict]


class DataStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_date: str | None
    latest_date: str | None
    total_records: int
    missing_count: int
    coverage_pct: float


class DataStatusResponse(BaseModel):
    trading_days: int
    start_date: str
    end_date: str
    stocks: list[DataStatusItem]


@router.post("/calendar", response_model=SyncCalendarResponse)
async def sync_calendar(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步交易日曆"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    new_count = await service.sync_trading_calendar(start_date, end_date)

    # 計算總數
    stmt = select(func.count()).select_from(TradingCalendar).where(
        TradingCalendar.date >= start_date,
        TradingCalendar.date <= end_date,
    )
    total = session.execute(stmt).scalar() or 0

    return SyncCalendarResponse(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        new_dates=new_count,
        total_dates=total,
    )


@router.post("/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的日K線"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_stock_daily(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/bulk", response_model=SyncBulkResponse)
async def sync_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場當日資料（TWSE RWD bulk）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_stock_daily_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/all", response_model=SyncAllResponse)
async def sync_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_all_stocks(start_date, end_date)

    return SyncAllResponse(
        stocks=result["stocks"],
        total_inserted=result["total_inserted"],
        errors=result["errors"],
    )


@router.get("/status", response_model=DataStatusResponse)
async def get_data_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得資料狀態"""
    if end_date is None:
        end_date = date.today()

    # 取得交易日數
    stmt = select(func.count()).select_from(TradingCalendar).where(
        TradingCalendar.date >= start_date,
        TradingCalendar.date <= end_date,
        TradingCalendar.is_trading_day == True,
    )
    trading_days = session.execute(stmt).scalar() or 0

    # 取得股票池
    stmt = select(StockUniverse).order_by(StockUniverse.rank)
    universe = session.execute(stmt).scalars().all()

    stocks = []
    for stock in universe:
        # 統計該股票的資料
        stmt = select(
            func.min(StockDaily.date),
            func.max(StockDaily.date),
            func.count(),
        ).where(
            StockDaily.stock_id == stock.stock_id,
            StockDaily.date >= start_date,
            StockDaily.date <= end_date,
        )
        result = session.execute(stmt).fetchone()
        earliest, latest, total = result

        missing = max(0, trading_days - total) if trading_days > 0 else 0
        coverage = (total / trading_days * 100) if trading_days > 0 else 0

        stocks.append(
            DataStatusItem(
                stock_id=stock.stock_id,
                name=stock.name,
                rank=stock.rank,
                earliest_date=earliest.isoformat() if earliest else None,
                latest_date=latest.isoformat() if latest else None,
                total_records=total,
                missing_count=missing,
                coverage_pct=round(coverage, 1),
            )
        )

    return DataStatusResponse(
        trading_days=trading_days,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        stocks=stocks,
    )


# =========================================================================
# PER/PBR/殖利率
# =========================================================================


@router.get("/per/status", response_model=DataStatusResponse)
async def get_per_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得 PER/PBR/殖利率 資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_per_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/per/bulk", response_model=SyncBulkResponse)
async def sync_per_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場 PER/PBR/殖利率（TWSE RWD bulk）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_per_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/per/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_per_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的 PER/PBR/殖利率（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_per(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/per/all", response_model=SyncAllResponse)
async def sync_per_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的 PER/PBR/殖利率（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)

    # 先同步交易日曆
    await service.sync_trading_calendar(start_date, end_date)

    # 取得股票池
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_per(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 三大法人買賣超
# =========================================================================


@router.get("/institutional/status", response_model=DataStatusResponse)
async def get_institutional_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得三大法人買賣超資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_institutional_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/institutional/bulk", response_model=SyncBulkResponse)
async def sync_institutional_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場三大法人買賣超（TWSE RWD）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_institutional_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/institutional/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_institutional_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的三大法人買賣超（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_institutional(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/institutional/all", response_model=SyncAllResponse)
async def sync_institutional_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的三大法人買賣超（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)

    # 先同步交易日曆
    await service.sync_trading_calendar(start_date, end_date)

    # 取得股票池
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_institutional(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 融資融券
# =========================================================================


@router.get("/margin/status", response_model=DataStatusResponse)
async def get_margin_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得融資融券資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_margin_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/margin/bulk", response_model=SyncBulkResponse)
async def sync_margin_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場融資融券（TWSE RWD）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_margin_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/margin/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_margin_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的融資融券（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_margin(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/margin/all", response_model=SyncAllResponse)
async def sync_margin_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的融資融券（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)

    # 先同步交易日曆
    await service.sync_trading_calendar(start_date, end_date)

    # 取得股票池
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_margin(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )
