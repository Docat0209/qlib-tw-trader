"""
交易記錄 API
"""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.portfolio import (
    TradeItem,
    TradesResponse,
)
from src.repositories.portfolio import TradeRepository

router = APIRouter()


# 股票名稱對照（簡易版，實際應從資料庫取得）
STOCK_NAMES = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2881": "富邦金",
}


@router.get("/trades", response_model=TradesResponse)
async def get_trades(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    """取得交易紀錄"""
    repo = TradeRepository(session)
    trades = repo.get_all(
        start_date=start_date,
        end_date=end_date,
        stock_id=symbol,
        limit=limit,
    )

    items = [
        TradeItem(
            id=trade.id,
            date=trade.date,
            symbol=trade.stock_id,
            name=STOCK_NAMES.get(trade.stock_id),
            side=trade.side,
            shares=trade.shares,
            price=float(trade.price),
            amount=float(trade.amount),
            commission=float(trade.commission),
            reason=trade.reason,
        )
        for trade in trades
    ]

    return TradesResponse(items=items, total=len(items))
