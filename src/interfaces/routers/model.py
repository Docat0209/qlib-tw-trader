"""
模型管理 API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.model import (
    ModelComparisonItem,
    ModelComparisonResponse,
    ModelHistoryItem,
    ModelHistoryResponse,
    ModelMetrics,
    ModelResponse,
    ModelStatus,
    Period,
    TrainRequest,
    TrainResponse,
)
from src.repositories.training import TrainingRepository

router = APIRouter()


def _run_to_response(run, factors: list[str]) -> ModelResponse:
    """轉換 TrainingRun 為 Response"""
    train_period = None
    valid_period = None

    if run.train_start and run.train_end:
        train_period = Period(start=run.train_start, end=run.train_end)
    if run.valid_start and run.valid_end:
        valid_period = Period(start=run.valid_start, end=run.valid_end)

    duration = None
    if run.started_at and run.completed_at:
        duration = int((run.completed_at - run.started_at).total_seconds())

    return ModelResponse(
        id=f"m{run.id:03d}",
        trained_at=run.completed_at or run.started_at,
        factor_count=run.factor_count,
        factors=factors,
        train_period=train_period,
        valid_period=valid_period,
        metrics=ModelMetrics(
            ic=float(run.model_ic) if run.model_ic else None,
            icir=float(run.icir) if run.icir else None,
        ),
        training_duration_seconds=duration,
        is_current=run.is_current,
    )


@router.get("/current", response_model=ModelResponse)
async def get_current_model(
    session: Session = Depends(get_db),
):
    """取得當前 active 模型"""
    repo = TrainingRepository(session)
    run = repo.get_current()

    if not run:
        raise HTTPException(status_code=404, detail="No model found")

    # 取得選中的因子
    results = repo.get_selected_factors(run.id)
    factors = [r.factor.name for r in results]

    return _run_to_response(run, factors)


@router.get("/history", response_model=ModelHistoryResponse)
async def get_model_history(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_db),
):
    """取得歷史模型 metadata"""
    repo = TrainingRepository(session)
    runs = repo.get_history(limit=limit)

    items = []
    for run in runs:
        train_period = None
        valid_period = None

        if run.train_start and run.train_end:
            train_period = Period(start=run.train_start, end=run.train_end)
        if run.valid_start and run.valid_end:
            valid_period = Period(start=run.valid_start, end=run.valid_end)

        items.append(
            ModelHistoryItem(
                id=f"m{run.id:03d}",
                trained_at=run.completed_at or run.started_at,
                factor_count=run.factor_count,
                train_period=train_period,
                valid_period=valid_period,
                metrics=ModelMetrics(
                    ic=float(run.model_ic) if run.model_ic else None,
                    icir=float(run.icir) if run.icir else None,
                ),
                is_current=run.is_current,
            )
        )

    return ModelHistoryResponse(items=items, total=len(items))


@router.get("/comparison", response_model=ModelComparisonResponse)
async def get_model_comparison(
    limit: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_db),
):
    """取得模型指標比較（用於圖表）"""
    repo = TrainingRepository(session)
    runs = repo.get_history(limit=limit)

    models = []
    for run in runs:
        trained_at = run.completed_at or run.started_at
        models.append(
            ModelComparisonItem(
                id=f"m{run.id:03d}",
                trained_at=trained_at.strftime("%Y-%m-%d") if trained_at else "",
                ic=float(run.model_ic) if run.model_ic else None,
                icir=float(run.icir) if run.icir else None,
                factor_count=run.factor_count,
            )
        )

    return ModelComparisonResponse(models=models)


@router.get("/status", response_model=ModelStatus)
async def get_training_status(
    session: Session = Depends(get_db),
):
    """取得訓練狀態（用於檢查是否需要重訓）"""
    repo = TrainingRepository(session)
    status = repo.get_status()

    return ModelStatus(**status)


@router.post("/train", response_model=TrainResponse)
async def trigger_training(
    data: TrainRequest,
    session: Session = Depends(get_db),
):
    """觸發模型訓練（非同步）"""
    # TODO: 實作非同步訓練
    # 目前只建立 queued 狀態的記錄
    repo = TrainingRepository(session)

    # 計算訓練期間（train_end 往前推）
    from datetime import timedelta
    train_start = data.train_end - timedelta(days=365 * 5)  # 5 年
    valid_start = data.train_end + timedelta(days=1)

    run = repo.create_run(
        train_start=train_start,
        train_end=data.train_end,
        valid_start=valid_start,
        valid_end=data.valid_end,
    )

    return TrainResponse(
        job_id=f"train_{run.id}",
        status="queued",
        message="訓練任務已排入佇列",
    )
