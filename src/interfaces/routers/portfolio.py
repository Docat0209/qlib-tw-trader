"""
交易記錄與預測 API
"""

import asyncio
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.portfolio import (
    PredictionRequest,
    PredictionSignal,
    PredictionsResponse,
    TradeItem,
    TradesResponse,
)
from src.repositories.models import StockDaily, StockUniverse
from src.repositories.portfolio import TradeRepository
from src.repositories.training import TrainingRepository
from src.services.predictor import Predictor
from src.services.qlib_exporter import ExportConfig, QlibExporter

router = APIRouter()

QLIB_DATA_DIR = Path("data/qlib")


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

    # 取得股票名稱
    stock_names = {s.stock_id: s.name for s in session.query(StockUniverse).all()}

    items = [
        TradeItem(
            id=trade.id,
            date=trade.date,
            symbol=trade.stock_id,
            name=stock_names.get(trade.stock_id),
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


@router.post("/predictions/generate", response_model=PredictionsResponse)
async def generate_predictions(
    request: PredictionRequest,
    session: Session = Depends(get_db),
):
    """執行預測並返回 Top K 股票"""
    # 驗證模型存在
    training_repo = TrainingRepository(session)
    model = training_repo.get_by_id(request.model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.name:
        raise HTTPException(status_code=400, detail="Model has no name")

    if model.status != "completed":
        raise HTTPException(status_code=400, detail="Model is not completed")

    # 決定預測日期
    if request.target_date:
        target_date = request.target_date
    else:
        # 使用資料庫中最新資料日期
        latest_date_row = session.query(func.max(StockDaily.date)).first()
        if not latest_date_row or not latest_date_row[0]:
            raise HTTPException(status_code=400, detail="No stock data available")
        target_date = latest_date_row[0]

    # 導出 qlib 資料（因子計算需要歷史資料）
    lookback_days = 180
    export_start = target_date - timedelta(days=lookback_days)

    def do_export():
        exporter = QlibExporter(session)
        exporter.export(ExportConfig(
            start_date=export_start,
            end_date=target_date,
            output_dir=QLIB_DATA_DIR,
        ))

    await asyncio.to_thread(do_export)

    # 執行預測
    def do_predict():
        predictor = Predictor(QLIB_DATA_DIR)
        return predictor.predict(
            model_name=model.name,
            target_date=target_date,
            top_k=request.top_k,
        )

    actual_date, signals = await asyncio.to_thread(do_predict)

    # 關聯股票名稱
    stock_names = {s.stock_id: s.name for s in session.query(StockUniverse).all()}

    response_signals = [
        PredictionSignal(
            rank=sig["rank"],
            symbol=sig["symbol"],
            name=stock_names.get(sig["symbol"]),
            score=round(sig["score"], 6),
        )
        for sig in signals
    ]

    return PredictionsResponse(
        date=actual_date.isoformat(),
        model_name=model.name,
        signals=response_signals,
    )
