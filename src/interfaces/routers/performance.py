"""
績效分析 API
"""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.performance import (
    Alpha,
    EquityCurvePoint,
    EquityCurveResponse,
    MonthlyReturn,
    MonthlyReturnsResponse,
    PerformanceSummary,
    Returns,
)

router = APIRouter()


@router.get("/summary", response_model=PerformanceSummary)
async def get_performance_summary(
    session: Session = Depends(get_db),
):
    """取得收益摘要"""
    # TODO: 從交易記錄計算實際收益
    # 目前返回空數據
    return PerformanceSummary(
        as_of=date.today().isoformat(),
        returns=Returns(
            today=None,
            wtd=None,
            mtd=None,
            ytd=None,
            total=None,
        ),
        benchmark_returns=Returns(
            today=None,
            wtd=None,
            mtd=None,
            ytd=None,
            total=None,
        ),
        alpha=Alpha(
            mtd=None,
            ytd=None,
        ),
    )


@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    benchmark: bool = Query(False, description="是否包含大盤"),
    session: Session = Depends(get_db),
):
    """取得權益曲線"""
    # TODO: 從交易記錄計算實際權益曲線
    # 目前返回空數據
    return EquityCurveResponse(data=[])


@router.get("/monthly", response_model=MonthlyReturnsResponse)
async def get_monthly_returns(
    session: Session = Depends(get_db),
):
    """取得月報酬"""
    # TODO: 從交易記錄計算實際月報酬
    # 目前返回空數據
    return MonthlyReturnsResponse(data=[])
