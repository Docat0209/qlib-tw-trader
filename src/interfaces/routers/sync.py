"""
資料同步 API
"""

import os
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.repositories.models import StockDaily, StockUniverse, TradingCalendar
from src.services.sync_service import SyncService

router = APIRouter()


@router.get("/debug/token")
async def debug_token():
    """調試：檢查 FINMIND_KEY 是否載入"""
    token = os.getenv("FINMIND_KEY", "")
    return {
        "token_loaded": bool(token),
        "token_length": len(token),
        "token_prefix": token[:20] + "..." if token else None,
    }


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


# 月營收專用 schema（用 year/month 而非 date）
class MonthlyStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int
    missing_months: list[str]


class MonthlyStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_month: str | None
    latest_month: str | None
    total_records: int
    missing_count: int
    coverage_pct: float


class MonthlyStatusResponse(BaseModel):
    expected_months: int
    start_year: int
    end_year: int
    stocks: list[MonthlyStatusItem]


# 季度財報專用 schema
class QuarterlyStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int
    missing_quarters: list[str]


class QuarterlyStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_quarter: str | None
    latest_quarter: str | None
    total_records: int
    missing_count: int
    coverage_pct: float


class QuarterlyStatusResponse(BaseModel):
    expected_quarters: int
    start_year: int
    end_year: int
    stocks: list[QuarterlyStatusItem]


# 股利政策專用 schema
class DividendStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int


class DividendStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_date: str | None
    latest_date: str | None
    total_records: int
    years_with_data: list[int]
    missing_years: list[int]


class DividendStatusResponse(BaseModel):
    start_year: int
    end_year: int
    stocks: list[DividendStatusItem]


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

    service = SyncService(session)

    # 取得整體交易日數（用於顯示）
    trading_days = service.count_trading_days(start_date, end_date)

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

        # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
        if earliest:
            expected_days = service.count_trading_days(earliest, end_date)
            missing = max(0, expected_days - total)
            coverage = SyncService._calc_coverage(total, expected_days, SyncService.MIN_DAILY_RECORDS)
        else:
            missing = 0
            coverage = 0

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


# =========================================================================
# 還原股價 (yfinance)
# =========================================================================


@router.get("/adj/status", response_model=DataStatusResponse)
async def get_adj_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得還原股價資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_adj_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/adj/bulk", response_model=SyncBulkResponse)
async def sync_adj_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場還原股價（yfinance 批次）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_adj_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/adj/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_adj_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的還原股價（yfinance）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_adj(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/adj/all", response_model=SyncAllResponse)
async def sync_adj_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的還原股價（yfinance）"""
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
            result = await service.sync_adj(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 外資持股
# =========================================================================


@router.get("/shareholding/status", response_model=DataStatusResponse)
async def get_shareholding_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得外資持股資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_shareholding_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/shareholding/bulk", response_model=SyncBulkResponse)
async def sync_shareholding_bulk(
    target_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步全市場外資持股（TWSE RWD）"""
    if target_date is None:
        target_date = date.today()

    service = SyncService(session)
    result = await service.sync_shareholding_bulk(target_date)

    return SyncBulkResponse(
        date=result["date"],
        total=result["total"],
        inserted=result["inserted"],
        error=result.get("error"),
    )


@router.post("/shareholding/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_shareholding_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的外資持股（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_shareholding(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/shareholding/all", response_model=SyncAllResponse)
async def sync_shareholding_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的外資持股（FinMind）"""
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
            result = await service.sync_shareholding(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 借券明細
# =========================================================================


@router.get("/securities-lending/status", response_model=DataStatusResponse)
async def get_securities_lending_status(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得借券明細資料狀態"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = service.get_securities_lending_status(start_date, end_date)

    return DataStatusResponse(
        trading_days=result["trading_days"],
        start_date=result["start_date"],
        end_date=result["end_date"],
        stocks=[DataStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/securities-lending/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_securities_lending_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的借券明細（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_securities_lending(stock_id, start_date, end_date)

    return SyncStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_dates=result["missing_dates"],
    )


@router.post("/securities-lending/all", response_model=SyncAllResponse)
async def sync_securities_lending_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的借券明細（FinMind）"""
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
            result = await service.sync_securities_lending(stock_id, start_date, end_date)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 月營收（低頻）
# =========================================================================


@router.get("/monthly-revenue/status", response_model=MonthlyStatusResponse)
async def get_monthly_revenue_status(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得月營收資料狀態"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = service.get_monthly_revenue_status(start_year, end_year)

    return MonthlyStatusResponse(
        expected_months=result["expected_months"],
        start_year=result["start_year"],
        end_year=result["end_year"],
        stocks=[MonthlyStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/monthly-revenue/stock/{stock_id}", response_model=MonthlyStockResponse)
async def sync_monthly_revenue_stock(
    stock_id: str,
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的月營收（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = await service.sync_monthly_revenue(stock_id, start_year, end_year)

    return MonthlyStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_months=result["missing_months"],
    )


@router.post("/monthly-revenue/all", response_model=SyncAllResponse)
async def sync_monthly_revenue_all(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的月營收（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)

    # 取得股票池
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_monthly_revenue(stock_id, start_year, end_year)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 季度財報 - 綜合損益表
# =========================================================================


@router.get("/financial/status", response_model=QuarterlyStatusResponse)
async def get_financial_status(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得綜合損益表資料狀態"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = service.get_quarterly_financial_status(start_year, end_year)

    return QuarterlyStatusResponse(
        expected_quarters=result["expected_quarters"],
        start_year=result["start_year"],
        end_year=result["end_year"],
        stocks=[QuarterlyStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/financial/stock/{stock_id}", response_model=QuarterlyStockResponse)
async def sync_financial_stock(
    stock_id: str,
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的綜合損益表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = await service.sync_quarterly_financial(stock_id, start_year, end_year)

    return QuarterlyStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_quarters=result["missing_quarters"],
    )


@router.post("/financial/all", response_model=SyncAllResponse)
async def sync_financial_all(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的綜合損益表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)

    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_quarterly_financial(stock_id, start_year, end_year)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 季度財報 - 資產負債表
# =========================================================================


@router.get("/balance/status", response_model=QuarterlyStatusResponse)
async def get_balance_status(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得資產負債表資料狀態"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = service.get_quarterly_balance_status(start_year, end_year)

    return QuarterlyStatusResponse(
        expected_quarters=result["expected_quarters"],
        start_year=result["start_year"],
        end_year=result["end_year"],
        stocks=[QuarterlyStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/balance/stock/{stock_id}", response_model=QuarterlyStockResponse)
async def sync_balance_stock(
    stock_id: str,
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的資產負債表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = await service.sync_quarterly_balance(stock_id, start_year, end_year)

    return QuarterlyStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_quarters=result["missing_quarters"],
    )


@router.post("/balance/all", response_model=SyncAllResponse)
async def sync_balance_all(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的資產負債表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)

    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_quarterly_balance(stock_id, start_year, end_year)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 季度財報 - 現金流量表
# =========================================================================


@router.get("/cashflow/status", response_model=QuarterlyStatusResponse)
async def get_cashflow_status(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得現金流量表資料狀態"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = service.get_quarterly_cashflow_status(start_year, end_year)

    return QuarterlyStatusResponse(
        expected_quarters=result["expected_quarters"],
        start_year=result["start_year"],
        end_year=result["end_year"],
        stocks=[QuarterlyStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/cashflow/stock/{stock_id}", response_model=QuarterlyStockResponse)
async def sync_cashflow_stock(
    stock_id: str,
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的現金流量表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = await service.sync_quarterly_cashflow(stock_id, start_year, end_year)

    return QuarterlyStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
        missing_quarters=result["missing_quarters"],
    )


@router.post("/cashflow/all", response_model=SyncAllResponse)
async def sync_cashflow_all(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的現金流量表（FinMind）"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)

    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    stock_ids = [row[0] for row in session.execute(stmt).fetchall()]

    total_inserted = 0
    errors = []

    for stock_id in stock_ids:
        try:
            result = await service.sync_quarterly_cashflow(stock_id, start_year, end_year)
            total_inserted += result["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock_id, "error": str(e)})

    return SyncAllResponse(
        stocks=len(stock_ids),
        total_inserted=total_inserted,
        errors=errors,
    )


# =========================================================================
# 股利政策 (Dividend)
# =========================================================================


@router.get("/dividend/status", response_model=DividendStatusResponse)
async def get_dividend_status(
    start_year: int = Query(default=2020),
    end_year: int = Query(default=None),
    session: Session = Depends(get_db),
):
    """取得股利資料狀態"""
    if end_year is None:
        end_year = date.today().year

    service = SyncService(session)
    result = service.get_dividend_status(start_year, end_year)

    return DividendStatusResponse(
        start_year=result["start_year"],
        end_year=result["end_year"],
        stocks=[DividendStatusItem(**s) for s in result["stocks"]],
    )


@router.post("/dividend/stock/{stock_id}", response_model=DividendStockResponse)
async def sync_dividend_stock(
    stock_id: str,
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步單一股票的股利資料（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_dividend(stock_id, start_date, end_date)

    return DividendStockResponse(
        stock_id=stock_id,
        fetched=result["fetched"],
        inserted=result["inserted"],
    )


@router.post("/dividend/all", response_model=SyncAllResponse)
async def sync_dividend_all(
    start_date: date = Query(default=date(2020, 1, 1)),
    end_date: date = Query(default=None),
    session: Session = Depends(get_db),
):
    """同步股票池內所有股票的股利資料（FinMind）"""
    if end_date is None:
        end_date = date.today()

    service = SyncService(session)
    result = await service.sync_dividend_all(start_date, end_date)

    # 轉換為統一的 SyncAllResponse 格式
    errors = [
        {"stock_id": s["stock_id"], "error": s.get("error", "")}
        for s in result["stocks"]
        if s["status"] == "error"
    ]

    return SyncAllResponse(
        stocks=result["total_stocks"],
        total_inserted=result["total_inserted"],
        errors=errors,
    )