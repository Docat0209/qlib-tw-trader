"""
模型管理 API
"""

import json
import shutil
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.model import (
    FactorSummary,
    ModelComparisonItem,
    ModelComparisonResponse,
    ModelHistoryResponse,
    ModelListResponse,
    ModelMetrics,
    ModelResponse,
    ModelStatus,
    ModelSummary,
    Period,
    TrainRequest,
    TrainResponse,
)
from src.repositories.factor import FactorRepository
from src.repositories.training import TrainingRepository
from src.shared.constants import TRAIN_DAYS, VALID_DAYS

router = APIRouter()

# 模型檔案目錄
MODELS_DIR = Path("data/models")


def _parse_factor_ids(json_str: str | None) -> list[int]:
    """解析 JSON 格式的因子 ID 列表"""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []


def _run_to_summary(run) -> ModelSummary:
    """轉換 TrainingRun 為 ModelSummary"""
    train_period = None
    valid_period = None

    if run.train_start and run.train_end:
        train_period = Period(start=run.train_start, end=run.train_end)
    if run.valid_start and run.valid_end:
        valid_period = Period(start=run.valid_start, end=run.valid_end)

    candidate_ids = _parse_factor_ids(run.candidate_factor_ids)
    selected_ids = _parse_factor_ids(run.selected_factor_ids)

    return ModelSummary(
        id=f"m{run.id:03d}",
        name=run.name,
        status=run.status,
        trained_at=run.completed_at or run.started_at,
        train_period=train_period,
        valid_period=valid_period,
        metrics=ModelMetrics(
            ic=float(run.model_ic) if run.model_ic else None,
            icir=float(run.icir) if run.icir else None,
        ),
        factor_count=run.factor_count or len(selected_ids),
        candidate_count=len(candidate_ids) if candidate_ids else None,
        is_current=run.is_current,
    )


def _run_to_response(
    run,
    factors: list[str],
    candidate_factors: list[FactorSummary] | None = None,
    selected_factors: list[FactorSummary] | None = None,
) -> ModelResponse:
    """轉換 TrainingRun 為 ModelResponse"""
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
        name=run.name,
        description=run.description,
        status=run.status,
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
        candidate_factors=candidate_factors or [],
        selected_factors=selected_factors or [],
    )


def _parse_model_id(model_id: str) -> int:
    """解析模型 ID (m001 -> 1)"""
    if model_id.startswith("m"):
        try:
            return int(model_id[1:])
        except ValueError:
            pass
    try:
        return int(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")


# === 端點定義 ===
# 注意：固定路由（/current, /status 等）必須放在動態路由（/{model_id}）之前


@router.get("", response_model=ModelListResponse)
async def list_models(
    session: Session = Depends(get_db),
):
    """取得所有模型列表"""
    repo = TrainingRepository(session)
    runs = repo.get_all()

    items = [_run_to_summary(run) for run in runs]
    return ModelListResponse(items=items, total=len(items))


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

    items = [_run_to_summary(run) for run in runs]
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
    from datetime import date as date_type

    repo = TrainingRepository(session)
    factor_repo = FactorRepository(session)

    # 計算日期
    today = date_type.today()
    valid_end = today
    train_end = data.train_end or (today - timedelta(days=VALID_DAYS))

    # 訓練期間
    train_start = train_end - timedelta(days=TRAIN_DAYS)
    valid_start = train_end + timedelta(days=1)

    # 生成模型名稱
    name = f"{train_start.strftime('%Y-%m')}~{train_end.strftime('%Y-%m')}"

    # 取得啟用的因子
    enabled_factors = factor_repo.get_all(enabled=True)
    candidate_ids = [f.id for f in enabled_factors]

    run = repo.create_run(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    # 更新 name 和 candidate_factor_ids
    run.name = name
    run.candidate_factor_ids = json.dumps(candidate_ids)
    repo._session.commit()

    # TODO: 實作非同步訓練任務
    return TrainResponse(
        job_id=f"train_{run.id}",
        status="queued",
        message=f"訓練任務已排入佇列 ({name})",
    )


# === 動態路由（必須放在固定路由之後） ===


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    session: Session = Depends(get_db),
):
    """取得單一模型詳情"""
    run_id = _parse_model_id(model_id)
    training_repo = TrainingRepository(session)
    factor_repo = FactorRepository(session)

    run = training_repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")

    # 取得所有因子結果
    all_results = training_repo.get_all_factor_results(run_id)

    # 建立因子 ID 到結果的映射
    result_map = {r.factor_id: r for r in all_results}

    # 取得候選因子列表
    candidate_ids = _parse_factor_ids(run.candidate_factor_ids)
    selected_ids = _parse_factor_ids(run.selected_factor_ids)

    candidate_factors = []
    selected_factors = []
    factor_names = []

    # 如果有候選因子 ID，使用它們
    if candidate_ids:
        for fid in candidate_ids:
            factor = factor_repo.get_by_id(fid)
            if factor:
                result = result_map.get(fid)
                summary = FactorSummary(
                    id=f"f{factor.id:03d}",
                    name=factor.name,
                    display_name=factor.display_name,
                    category=factor.category,
                    ic_value=float(result.ic_value) if result else None,
                )
                candidate_factors.append(summary)
                if fid in selected_ids:
                    selected_factors.append(summary)
                    factor_names.append(factor.name)
    else:
        # 從 TrainingFactorResult 取得（向後兼容）
        for result in all_results:
            factor = result.factor
            summary = FactorSummary(
                id=f"f{factor.id:03d}",
                name=factor.name,
                display_name=factor.display_name,
                category=factor.category,
                ic_value=float(result.ic_value),
            )
            candidate_factors.append(summary)
            if result.selected:
                selected_factors.append(summary)
                factor_names.append(factor.name)

    return _run_to_response(run, factor_names, candidate_factors, selected_factors)


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    session: Session = Depends(get_db),
):
    """刪除模型"""
    run_id = _parse_model_id(model_id)
    repo = TrainingRepository(session)

    run = repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")

    if run.is_current:
        raise HTTPException(status_code=400, detail="Cannot delete current model")

    # 刪除模型檔案目錄
    if run.name:
        model_dir = MODELS_DIR / run.name
        if model_dir.exists():
            shutil.rmtree(model_dir)

    # 刪除資料庫記錄
    repo.delete(run_id)

    return {"status": "deleted", "id": model_id}


@router.patch("/{model_id}/current")
async def set_current_model(
    model_id: str,
    session: Session = Depends(get_db),
):
    """設為當前模型"""
    run_id = _parse_model_id(model_id)
    repo = TrainingRepository(session)

    run = repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")

    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Model is not completed")

    repo.set_current(run_id)

    return {"status": "success", "id": model_id, "is_current": True}
