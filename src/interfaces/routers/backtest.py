"""
回測 API
"""

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.backtest import (
    BacktestDetailResponse,
    BacktestListResponse,
    BacktestMetrics,
    BacktestRequest,
    BacktestResponse,
    BacktestRunResponse,
    EquityCurvePoint,
    KlinePoint,
    StockKlineResponse,
    StockTradeInfo,
    StockTradeListResponse,
    TradePoint,
)
from src.repositories.backtest import BacktestRepository
from src.repositories.training import TrainingRepository
from src.services.job_manager import job_manager

router = APIRouter()

# qlib 資料目錄
QLIB_DATA_DIR = Path("data/qlib")


def backtest_to_response(bt) -> BacktestResponse:
    """轉換 Backtest 為 Response"""
    metrics = None
    if bt.result:
        try:
            result_data = json.loads(bt.result)
            if "error" not in result_data:
                metrics = BacktestMetrics(
                    # 新格式
                    total_return_with_cost=result_data.get("total_return_with_cost"),
                    total_return_without_cost=result_data.get("total_return_without_cost"),
                    annual_return_with_cost=result_data.get("annual_return_with_cost"),
                    annual_return_without_cost=result_data.get("annual_return_without_cost"),
                    sharpe_ratio=result_data.get("sharpe_ratio"),
                    max_drawdown=result_data.get("max_drawdown"),
                    win_rate=result_data.get("win_rate"),
                    total_trades=result_data.get("total_trades"),
                    total_cost=result_data.get("total_cost"),
                    # 向後兼容舊格式
                    total_return=result_data.get("total_return") or result_data.get("total_return_with_cost"),
                    annual_return=result_data.get("annual_return") or result_data.get("annual_return_with_cost"),
                    profit_factor=result_data.get("profit_factor"),
                )
        except json.JSONDecodeError:
            pass

    return BacktestResponse(
        id=bt.id,
        model_id=bt.model_id,
        start_date=bt.start_date.isoformat(),
        end_date=bt.end_date.isoformat(),
        initial_capital=float(bt.initial_capital),
        max_positions=bt.max_positions,
        status=bt.status,
        metrics=metrics,
        created_at=bt.created_at.isoformat() if bt.created_at else "",
    )


@router.get("", response_model=BacktestListResponse)
async def list_backtests(
    session: Session = Depends(get_db),
    model_id: int | None = None,
    limit: int = 20,
):
    """取得回測列表"""
    repo = BacktestRepository(session)

    if model_id:
        backtests = repo.get_by_model(model_id, limit)
    else:
        backtests = repo.get_recent(limit)

    return BacktestListResponse(
        items=[backtest_to_response(bt) for bt in backtests],
        total=len(backtests),
    )


@router.get("/{backtest_id}", response_model=BacktestDetailResponse)
async def get_backtest(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """取得回測詳情"""
    repo = BacktestRepository(session)
    bt = repo.get(backtest_id)

    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # Parse metrics
    metrics = None
    if bt.result:
        try:
            result_data = json.loads(bt.result)
            if "error" not in result_data:
                metrics = BacktestMetrics(
                    # 新格式
                    total_return_with_cost=result_data.get("total_return_with_cost"),
                    total_return_without_cost=result_data.get("total_return_without_cost"),
                    annual_return_with_cost=result_data.get("annual_return_with_cost"),
                    annual_return_without_cost=result_data.get("annual_return_without_cost"),
                    sharpe_ratio=result_data.get("sharpe_ratio"),
                    max_drawdown=result_data.get("max_drawdown"),
                    win_rate=result_data.get("win_rate"),
                    total_trades=result_data.get("total_trades"),
                    total_cost=result_data.get("total_cost"),
                    # 向後兼容舊格式
                    total_return=result_data.get("total_return") or result_data.get("total_return_with_cost"),
                    annual_return=result_data.get("annual_return") or result_data.get("annual_return_with_cost"),
                    profit_factor=result_data.get("profit_factor"),
                )
        except json.JSONDecodeError:
            pass

    # Parse equity curve
    equity_curve = None
    if bt.equity_curve:
        try:
            curve_data = json.loads(bt.equity_curve)
            equity_curve = [
                EquityCurvePoint(
                    date=p.get("date", ""),
                    equity=p.get("equity", 0),
                    benchmark=p.get("benchmark"),
                    drawdown=p.get("drawdown"),
                )
                for p in curve_data
            ]
        except json.JSONDecodeError:
            pass

    return BacktestDetailResponse(
        id=bt.id,
        model_id=bt.model_id,
        start_date=bt.start_date.isoformat(),
        end_date=bt.end_date.isoformat(),
        initial_capital=float(bt.initial_capital),
        max_positions=bt.max_positions,
        status=bt.status,
        metrics=metrics,
        equity_curve=equity_curve,
        created_at=bt.created_at.isoformat() if bt.created_at else "",
    )


@router.post("/run", response_model=BacktestRunResponse)
async def run_backtest(
    request: BacktestRequest,
    session: Session = Depends(get_db),
):
    """執行回測"""
    # 驗證模型存在
    training_repo = TrainingRepository(session)
    model = training_repo.get_by_id(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # 從模型取得回測期間
    # 回測期間 = valid_end + 1 天 ~ valid_end + 1 個月
    if not model.valid_end:
        raise HTTPException(status_code=400, detail="Model has no validation period defined")

    start_date = model.valid_end + timedelta(days=1)
    end_date = start_date + relativedelta(months=1)

    # 若 end_date 超過今天，則用今天
    if end_date > date.today():
        end_date = date.today()

    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="No backtest period available")

    # 建立回測記錄
    backtest_repo = BacktestRepository(session)
    backtest = backtest_repo.create(
        model_id=request.model_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(request.initial_capital)),
        max_positions=request.max_positions,
    )

    # 建立非同步任務
    job_id = await job_manager.create_job(
        job_type="backtest",
        task_fn=run_backtest_task,
        message=f"Running backtest {backtest.id}",
        backtest_id=backtest.id,
        model_id=request.model_id,
        model_name=model.name,  # 模型名稱用於載入檔案
        start_date=start_date,
        end_date=end_date,
        initial_capital=request.initial_capital,
        max_positions=request.max_positions,
    )

    return BacktestRunResponse(
        backtest_id=backtest.id,
        job_id=job_id,
        status="queued",
        message="Backtest started",
    )


async def run_backtest_task(
    progress_callback,
    backtest_id: int,
    model_id: int,
    model_name: str,
    start_date: date,
    end_date: date,
    initial_capital: float,
    max_positions: int,
):
    """
    回測任務 - 使用 Backtester 服務
    """
    import asyncio
    from src.repositories.backtest import BacktestRepository
    from src.repositories.database import get_session
    from src.services.backtester import Backtester
    from src.services.qlib_exporter import QlibExporter, ExportConfig

    session = get_session()
    repo = BacktestRepository(session)

    loop = asyncio.get_event_loop()

    # 同步回調轉非同步（在主線程安全調用）
    def sync_progress(progress: float, message: str):
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(progress_callback(progress, message))
        )

    try:
        # 更新狀態
        repo.update_status(backtest_id, "running")
        await progress_callback(1, "Preparing qlib data export...")

        # 匯出回測期間的 qlib 資料（避免 lookahead bias）
        # 需要額外的回溯期給因子計算（如 126 天均線）
        lookback_days = 180  # 6 個月緩衝
        export_start = start_date - timedelta(days=lookback_days)

        await progress_callback(2, f"Exporting data: {export_start} ~ {end_date}...")

        # 在執行緒中執行同步的匯出操作（避免阻塞事件循環）
        def do_export():
            export_session = get_session()
            try:
                exporter = QlibExporter(export_session)
                export_config = ExportConfig(
                    start_date=export_start,
                    end_date=end_date,
                    output_dir=QLIB_DATA_DIR,
                )
                exporter.export(export_config)
            finally:
                export_session.close()

        await asyncio.to_thread(do_export)

        await progress_callback(4, "Qlib data ready, starting backtest...")

        # 在執行緒中執行同步的回測操作（帶進度回調）
        def do_backtest():
            backtester = Backtester(QLIB_DATA_DIR)
            return backtester.run(
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                topk=max_positions,
                n_drop=1,
                on_progress=sync_progress,  # 使用線程安全的回調
            )

        result = await asyncio.to_thread(do_backtest)

        await progress_callback(96, "Saving results to database...")

        # 轉換結果格式
        metrics_dict = {
            "total_return_with_cost": result.metrics.total_return_with_cost,
            "total_return_without_cost": result.metrics.total_return_without_cost,
            "annual_return_with_cost": result.metrics.annual_return_with_cost,
            "annual_return_without_cost": result.metrics.annual_return_without_cost,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "max_drawdown": result.metrics.max_drawdown,
            "win_rate": result.metrics.win_rate,
            "total_trades": result.metrics.total_trades,
            "total_cost": result.metrics.total_cost,
        }

        equity_curve = [
            {
                "date": p.date,
                "equity": p.equity,
                "benchmark": p.benchmark,
                "drawdown": p.drawdown,
            }
            for p in result.equity_curve
        ]

        # 儲存結果
        await progress_callback(97, "Saving metrics and equity curve...")
        repo.complete(backtest_id, metrics_dict, equity_curve)

        # 儲存交易記錄
        if result.trades:
            await progress_callback(98, f"Saving {len(result.trades)} trade records...")
            from src.repositories.models import Trade
            from decimal import Decimal

            for t in result.trades:
                trade = Trade(
                    backtest_id=backtest_id,
                    date=t.date,
                    stock_id=t.stock_id,
                    side=t.side,
                    shares=t.shares,
                    price=Decimal(str(t.price)),
                    amount=Decimal(str(t.amount)),
                    commission=Decimal(str(t.commission)),
                    reason="backtest",
                )
                session.add(trade)
            session.commit()

        await progress_callback(100, f"Backtest completed! Return: {metrics_dict['total_return_with_cost']:.2f}%")

        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "metrics": metrics_dict,
        }

    except Exception as e:
        repo.fail(backtest_id, str(e))
        raise

    finally:
        session.close()


# === 股票交易 API ===


@router.get("/{backtest_id}/stocks", response_model=StockTradeListResponse)
async def get_backtest_stocks(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """取得回測中交易過的股票清單"""
    from sqlalchemy import func, case
    from src.repositories.models import Trade

    # 驗證回測存在
    repo = BacktestRepository(session)
    bt = repo.get(backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # 查詢交易記錄統計
    trades = (
        session.query(
            Trade.stock_id,
            func.sum(case((Trade.side == "buy", 1), else_=0)).label("buy_count"),
            func.sum(case((Trade.side == "sell", 1), else_=0)).label("sell_count"),
        )
        .filter(Trade.backtest_id == backtest_id)
        .group_by(Trade.stock_id)
        .all()
    )

    # 取得股票名稱（從 stock_universe）
    from src.repositories.models import StockUniverse

    stock_names = {
        s.stock_id: s.name
        for s in session.query(StockUniverse).all()
    }

    items = [
        StockTradeInfo(
            stock_id=t.stock_id,
            name=stock_names.get(t.stock_id, t.stock_id),
            buy_count=t.buy_count or 0,
            sell_count=t.sell_count or 0,
            total_pnl=None,  # 需要更多計算
        )
        for t in trades
    ]

    return StockTradeListResponse(
        backtest_id=backtest_id,
        items=items,
        total=len(items),
    )


@router.get("/{backtest_id}/stocks/{stock_id}", response_model=StockKlineResponse)
async def get_stock_kline(
    backtest_id: int,
    stock_id: str,
    session: Session = Depends(get_db),
):
    """取得個股 K 線 + 買賣點"""
    from src.repositories.models import Trade, StockDaily, StockUniverse

    # 驗證回測存在
    repo = BacktestRepository(session)
    bt = repo.get(backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # 取得股票名稱
    stock = session.query(StockUniverse).filter(
        StockUniverse.stock_id == stock_id
    ).first()
    stock_name = stock.name if stock else stock_id

    # 取得 K 線資料
    klines_raw = (
        session.query(StockDaily)
        .filter(
            StockDaily.stock_id == stock_id,
            StockDaily.date >= bt.start_date,
            StockDaily.date <= bt.end_date,
        )
        .order_by(StockDaily.date)
        .all()
    )

    klines = [
        KlinePoint(
            date=k.date.isoformat(),
            open=float(k.open),
            high=float(k.high),
            low=float(k.low),
            close=float(k.close),
            volume=k.volume,
        )
        for k in klines_raw
    ]

    # 取得交易記錄
    trades_raw = (
        session.query(Trade)
        .filter(
            Trade.backtest_id == backtest_id,
            Trade.stock_id == stock_id,
        )
        .order_by(Trade.date)
        .all()
    )

    trades = [
        TradePoint(
            date=t.date.isoformat(),
            side=t.side,
            price=float(t.price),
            shares=t.shares,
        )
        for t in trades_raw
    ]

    return StockKlineResponse(
        stock_id=stock_id,
        name=stock_name,
        klines=klines,
        trades=trades,
    )
