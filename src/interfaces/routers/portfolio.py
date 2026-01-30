"""
持倉/交易/預測 API
"""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.portfolio import (
    PositionItem,
    PositionsResponse,
    PredictionHistoryItem,
    PredictionSignal,
    PredictionsHistoryResponse,
    PredictionsLatestResponse,
    TradeItem,
    TradesResponse,
)
from src.repositories.portfolio import (
    PositionRepository,
    PredictionRepository,
    TradeRepository,
)

router = APIRouter()


# 股票名稱對照（簡易版，實際應從資料庫取得）
STOCK_NAMES = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2881": "富邦金",
}


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(
    session: Session = Depends(get_db),
):
    """取得當前持倉"""
    repo = PositionRepository(session)
    positions = repo.get_all()

    # TODO: 取得當前價格（從 OHLCV 最新資料）
    # 目前先使用 avg_cost 作為 current_price

    items = []
    total_value = 0.0

    for pos in positions:
        current_price = float(pos.avg_cost)  # TODO: 使用實際當前價格
        market_value = current_price * pos.shares
        unrealized_pnl = market_value - float(pos.avg_cost) * pos.shares
        unrealized_pnl_pct = unrealized_pnl / (float(pos.avg_cost) * pos.shares) if pos.shares > 0 else 0

        total_value += market_value

        items.append(
            PositionItem(
                symbol=pos.stock_id,
                name=STOCK_NAMES.get(pos.stock_id),
                shares=pos.shares,
                avg_cost=float(pos.avg_cost),
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                weight=None,  # 計算後更新
            )
        )

    # 計算權重
    for item in items:
        if total_value > 0 and item.market_value:
            item.weight = item.market_value / total_value

    return PositionsResponse(
        as_of=date.today(),
        total_value=total_value,
        cash=0.0,  # TODO: 從某處取得現金餘額
        positions=items,
    )


@router.get("/predictions/latest", response_model=PredictionsLatestResponse)
async def get_latest_predictions(
    session: Session = Depends(get_db),
):
    """取得最新預測信號"""
    repo = PredictionRepository(session)
    pos_repo = PositionRepository(session)
    predictions = repo.get_latest()

    if not predictions:
        return PredictionsLatestResponse(
            date=date.today(),
            model_id="---",
            signals=[],
        )

    signals = []
    for pred in predictions:
        position = pos_repo.get_by_stock(pred.stock_id)
        signals.append(
            PredictionSignal(
                symbol=pred.stock_id,
                name=STOCK_NAMES.get(pred.stock_id),
                score=float(pred.score),
                rank=pred.rank,
                signal=pred.signal,
                current_position=position.shares if position else 0,
            )
        )

    return PredictionsLatestResponse(
        date=predictions[0].date,
        model_id=f"m{predictions[0].model_id:03d}",
        signals=signals,
    )


@router.get("/predictions/history", response_model=PredictionsHistoryResponse)
async def get_prediction_history(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    """取得歷史預測紀錄"""
    repo = PredictionRepository(session)
    predictions = repo.get_history(
        start_date=start_date,
        end_date=end_date,
        stock_id=symbol,
        limit=limit,
    )

    items = [
        PredictionHistoryItem(
            date=pred.date,
            symbol=pred.stock_id,
            score=float(pred.score),
            rank=pred.rank,
            signal=pred.signal,
        )
        for pred in predictions
    ]

    return PredictionsHistoryResponse(items=items, total=len(items))


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
