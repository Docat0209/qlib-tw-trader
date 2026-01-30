"""
回測 API
"""

import json
from datetime import date
from decimal import Decimal

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
)
from src.repositories.backtest import BacktestRepository
from src.repositories.training import TrainingRepository
from src.services.job_manager import job_manager

router = APIRouter()


def backtest_to_response(bt) -> BacktestResponse:
    """轉換 Backtest 為 Response"""
    metrics = None
    if bt.result:
        try:
            result_data = json.loads(bt.result)
            if "error" not in result_data:
                metrics = BacktestMetrics(
                    total_return=result_data.get("total_return"),
                    annual_return=result_data.get("annual_return"),
                    sharpe_ratio=result_data.get("sharpe_ratio"),
                    max_drawdown=result_data.get("max_drawdown"),
                    win_rate=result_data.get("win_rate"),
                    profit_factor=result_data.get("profit_factor"),
                    total_trades=result_data.get("total_trades"),
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
                    total_return=result_data.get("total_return"),
                    annual_return=result_data.get("annual_return"),
                    sharpe_ratio=result_data.get("sharpe_ratio"),
                    max_drawdown=result_data.get("max_drawdown"),
                    win_rate=result_data.get("win_rate"),
                    profit_factor=result_data.get("profit_factor"),
                    total_trades=result_data.get("total_trades"),
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

    # 建立回測記錄
    backtest_repo = BacktestRepository(session)
    backtest = backtest_repo.create(
        model_id=request.model_id,
        start_date=request.start_date,
        end_date=request.end_date,
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
        start_date=request.start_date,
        end_date=request.end_date,
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
    start_date: date,
    end_date: date,
    initial_capital: float,
    max_positions: int,
):
    """
    回測任務

    這是一個模擬實作，實際回測邏輯需要整合 qlib
    """
    from src.repositories.backtest import BacktestRepository
    from src.repositories.database import get_session

    session = get_session()
    repo = BacktestRepository(session)

    try:
        # 更新狀態
        repo.update_status(backtest_id, "running")
        await progress_callback(10, "Loading data...")

        # 模擬回測過程
        import asyncio
        import random
        from datetime import timedelta

        await asyncio.sleep(1)
        await progress_callback(30, "Running simulation...")

        await asyncio.sleep(1)
        await progress_callback(60, "Calculating metrics...")

        await asyncio.sleep(1)
        await progress_callback(80, "Generating equity curve...")

        # 生成模擬結果
        total_days = (end_date - start_date).days
        equity = initial_capital
        peak = equity
        equity_curve = []

        current_date = start_date
        while current_date <= end_date:
            # 模擬每日收益
            daily_return = random.gauss(0.0005, 0.015)
            equity *= (1 + daily_return)
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak if peak > 0 else 0

            equity_curve.append({
                "date": current_date.isoformat(),
                "equity": round(equity, 2),
                "benchmark": round(initial_capital * (1 + 0.0003 * (current_date - start_date).days), 2),
                "drawdown": round(drawdown * 100, 2),
            })
            current_date += timedelta(days=1)

        # 計算績效指標
        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
        total_return = (final_equity - initial_capital) / initial_capital
        years = total_days / 365.0
        annual_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
        max_dd = max(p["drawdown"] for p in equity_curve) if equity_curve else 0

        result = {
            "total_return": round(total_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "sharpe_ratio": round(random.uniform(0.5, 2.0), 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(random.uniform(45, 65), 1),
            "profit_factor": round(random.uniform(1.0, 2.0), 2),
            "total_trades": random.randint(50, 200),
        }

        await progress_callback(95, "Saving results...")

        # 儲存結果
        repo.complete(backtest_id, result, equity_curve)

        await progress_callback(100, "Backtest completed")

        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "metrics": result,
        }

    except Exception as e:
        repo.fail(backtest_id, str(e))
        raise

    finally:
        session.close()
