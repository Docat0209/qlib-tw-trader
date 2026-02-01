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
    AllTradesResponse,
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


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """刪除回測記錄（連同交易記錄）"""
    from src.repositories.models import Trade

    repo = BacktestRepository(session)
    bt = repo.get(backtest_id)

    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # 刪除相關的交易記錄
    session.query(Trade).filter(Trade.backtest_id == backtest_id).delete()

    # 刪除回測記錄
    session.delete(bt)
    session.commit()

    return {"message": "Backtest deleted", "id": backtest_id}


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

    if not model.name:
        raise HTTPException(status_code=400, detail="Model has no name")

    # 從模型目錄的 config.json 讀取 valid_end
    # 回測期間 = valid_end + 1 天 ~ valid_end + 1 個月
    model_dir = Path("data/models") / model.name
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise HTTPException(status_code=400, detail="Model config.json not found")

    with open(config_path) as f:
        config = json.load(f)

    valid_end_str = config.get("valid_end")
    if not valid_end_str:
        raise HTTPException(status_code=400, detail="Model config has no valid_end")

    valid_end = date.fromisoformat(valid_end_str)
    start_date = valid_end + timedelta(days=1)
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
        trade_price=request.trade_price,
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
    trade_price: str = "close",
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
                trade_price=trade_price,
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

    # 計算每檔股票的總盈虧
    def calculate_stock_pnl(stock_id: str) -> float | None:
        """FIFO 計算股票總盈虧"""
        stock_trades = (
            session.query(Trade)
            .filter(Trade.backtest_id == backtest_id, Trade.stock_id == stock_id)
            .order_by(Trade.date)
            .all()
        )

        if not stock_trades:
            return None

        buy_queue = []
        total_pnl = 0.0

        for t in stock_trades:
            if t.side == "buy":
                buy_queue.append({
                    "price": float(t.price),
                    "shares": t.shares,
                })
            else:
                # 賣出：FIFO 配對
                remaining = t.shares
                cost = 0.0
                while remaining > 0 and buy_queue:
                    buy = buy_queue[0]
                    if buy["shares"] <= remaining:
                        cost += buy["price"] * buy["shares"]
                        remaining -= buy["shares"]
                        buy_queue.pop(0)
                    else:
                        cost += buy["price"] * remaining
                        buy["shares"] -= remaining
                        remaining = 0

                sell_amount = float(t.price) * t.shares
                commission = float(t.commission) if t.commission else 0
                total_pnl += sell_amount - cost - commission

        return round(total_pnl, 2) if total_pnl != 0 else None

    items = [
        StockTradeInfo(
            stock_id=t.stock_id,
            name=stock_names.get(t.stock_id, t.stock_id),
            buy_count=t.buy_count or 0,
            sell_count=t.sell_count or 0,
            total_pnl=calculate_stock_pnl(t.stock_id),
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

    # FIFO 配對計算盈虧
    trades = []
    buy_queue = []  # 買入記錄佇列

    for t in trades_raw:
        trade_point = TradePoint(
            date=t.date.isoformat(),
            side=t.side,
            price=float(t.price),
            shares=t.shares,
            amount=float(t.amount) if t.amount else float(t.price) * t.shares,
            commission=float(t.commission) if t.commission else 0,
            pnl=None,
            pnl_pct=None,
            holding_days=None,
        )

        if t.side == "buy":
            # 記錄買入
            buy_queue.append({
                "date": t.date,
                "price": float(t.price),
                "shares": t.shares,
            })
        else:
            # 賣出：FIFO 配對計算盈虧
            remaining_shares = t.shares
            total_cost = 0.0
            earliest_buy_date = None

            while remaining_shares > 0 and buy_queue:
                buy = buy_queue[0]
                if earliest_buy_date is None:
                    earliest_buy_date = buy["date"]

                if buy["shares"] <= remaining_shares:
                    total_cost += buy["price"] * buy["shares"]
                    remaining_shares -= buy["shares"]
                    buy_queue.pop(0)
                else:
                    total_cost += buy["price"] * remaining_shares
                    buy["shares"] -= remaining_shares
                    remaining_shares = 0

            # 計算盈虧
            if t.shares > 0:
                sell_amount = float(t.price) * t.shares
                commission = float(t.commission) if t.commission else 0
                pnl = sell_amount - total_cost - commission
                avg_cost = total_cost / t.shares
                pnl_pct = ((float(t.price) / avg_cost) - 1) * 100 if avg_cost > 0 else 0

                trade_point.pnl = round(pnl, 2)
                trade_point.pnl_pct = round(pnl_pct, 2)

                if earliest_buy_date:
                    trade_point.holding_days = (t.date - earliest_buy_date).days

        trades.append(trade_point)

    return StockKlineResponse(
        stock_id=stock_id,
        name=stock_name,
        klines=klines,
        trades=trades,
    )


@router.get("/{backtest_id}/trades", response_model=AllTradesResponse)
async def get_all_trades(
    backtest_id: int,
    session: Session = Depends(get_db),
):
    """取得回測所有交易記錄（按日期排序，含 FIFO 盈虧計算）"""
    from src.repositories.models import Trade, StockUniverse

    # 驗證回測存在
    repo = BacktestRepository(session)
    bt = repo.get(backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # 取得所有交易記錄
    trades_raw = (
        session.query(Trade)
        .filter(Trade.backtest_id == backtest_id)
        .order_by(Trade.date, Trade.stock_id)
        .all()
    )

    # 取得股票名稱
    stock_names = {
        s.stock_id: s.name
        for s in session.query(StockUniverse).all()
    }

    # 按股票分組做 FIFO 配對
    stock_queues: dict[str, list] = {}
    trades = []
    total_pnl = 0.0

    for t in trades_raw:
        stock_id = t.stock_id
        if stock_id not in stock_queues:
            stock_queues[stock_id] = []

        trade_point = TradePoint(
            date=t.date.isoformat(),
            side=t.side,
            price=float(t.price),
            shares=t.shares,
            amount=float(t.amount) if t.amount else float(t.price) * t.shares,
            commission=float(t.commission) if t.commission else 0,
            pnl=None,
            pnl_pct=None,
            holding_days=None,
            stock_id=stock_id,
            stock_name=stock_names.get(stock_id, stock_id),
        )

        if t.side == "buy":
            # 記錄買入資訊，包含手續費
            buy_commission = float(t.commission) if t.commission else 0
            stock_queues[stock_id].append({
                "date": t.date,
                "price": float(t.price),
                "shares": t.shares,
                "commission": buy_commission,  # 買入手續費
            })
        else:
            # 賣出：FIFO 配對
            buy_queue = stock_queues[stock_id]
            remaining_shares = t.shares
            total_cost = 0.0
            total_buy_commission = 0.0
            earliest_buy_date = None

            while remaining_shares > 0 and buy_queue:
                buy = buy_queue[0]
                if earliest_buy_date is None:
                    earliest_buy_date = buy["date"]

                if buy["shares"] <= remaining_shares:
                    total_cost += buy["price"] * buy["shares"]
                    total_buy_commission += buy.get("commission", 0)
                    remaining_shares -= buy["shares"]
                    buy_queue.pop(0)
                else:
                    # 部分賣出：按比例分攤買入手續費
                    ratio = remaining_shares / buy["shares"]
                    total_cost += buy["price"] * remaining_shares
                    total_buy_commission += buy.get("commission", 0) * ratio
                    buy["shares"] -= remaining_shares
                    buy["commission"] = buy.get("commission", 0) * (1 - ratio)
                    remaining_shares = 0

            # 計算盈虧（含買入+賣出手續費）
            if t.shares > 0:
                sell_amount = float(t.price) * t.shares
                sell_commission = float(t.commission) if t.commission else 0
                pnl = sell_amount - total_cost - total_buy_commission - sell_commission
                avg_cost = (total_cost + total_buy_commission) / t.shares
                pnl_pct = ((float(t.price) / (total_cost / t.shares)) - 1) * 100 if total_cost > 0 else 0

                trade_point.pnl = round(pnl, 2)
                trade_point.pnl_pct = round(pnl_pct, 2)
                total_pnl += pnl

                if earliest_buy_date:
                    trade_point.holding_days = (t.date - earliest_buy_date).days

        trades.append(trade_point)

    # 按日期重新排序
    trades.sort(key=lambda x: x.date)

    # 計算未平倉部位的未實現盈虧
    from src.repositories.models import StockDaily

    unrealized_pnl = 0.0
    for stock_id, buy_queue in stock_queues.items():
        if buy_queue:  # 還有未賣出的買入記錄
            # 取得回測結束日的收盤價
            latest_price_row = (
                session.query(StockDaily.close)
                .filter(StockDaily.stock_id == stock_id)
                .filter(StockDaily.date <= bt.end_date)
                .order_by(StockDaily.date.desc())
                .first()
            )
            if latest_price_row:
                latest_price = float(latest_price_row[0])
                for buy in buy_queue:
                    # 未實現盈虧 = (當前價 - 買入價) × 股數 - 買入手續費
                    market_value = latest_price * buy["shares"]
                    cost = buy["price"] * buy["shares"] + buy.get("commission", 0)
                    unrealized_pnl += market_value - cost

    total_equity_pnl = total_pnl + unrealized_pnl

    return AllTradesResponse(
        backtest_id=backtest_id,
        items=trades,
        total_pnl=round(total_pnl, 2),
        unrealized_pnl=round(unrealized_pnl, 2),
        total_equity_pnl=round(total_equity_pnl, 2),
        total=len(trades),
    )
