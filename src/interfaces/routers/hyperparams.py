"""超參數管理 API"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.hyperparams import (
    CultivateRequest,
    CultivateResponse,
    CultivationPeriod,
    HyperparamsDetail,
    HyperparamsListResponse,
    HyperparamsSummary,
    HyperparamsUpdate,
)
from src.repositories.factor import FactorRepository
from src.repositories.hyperparams import HyperparamsRepository
from src.services.job_manager import broadcast_data_updated

router = APIRouter()


def _to_summary(hp) -> HyperparamsSummary:
    """轉換為摘要"""
    params = json.loads(hp.params_json)
    return HyperparamsSummary(
        id=hp.id,
        name=hp.name,
        cultivated_at=hp.cultivated_at,
        n_periods=hp.n_periods,
        learning_rate=params.get("learning_rate"),
        num_leaves=params.get("num_leaves"),
    )


def _to_detail(hp) -> HyperparamsDetail:
    """轉換為詳情"""
    params = json.loads(hp.params_json)
    stability = json.loads(hp.stability_json)
    periods_raw = json.loads(hp.periods_json)

    periods = [
        CultivationPeriod(
            train_start=p["train_start"],
            train_end=p["train_end"],
            valid_start=p["valid_start"],
            valid_end=p["valid_end"],
            best_ic=p["best_ic"],
            params=p["params"],
        )
        for p in periods_raw
    ]

    return HyperparamsDetail(
        id=hp.id,
        name=hp.name,
        cultivated_at=hp.cultivated_at,
        n_periods=hp.n_periods,
        params=params,
        stability=stability,
        periods=periods,
    )


@router.get("", response_model=HyperparamsListResponse)
async def list_hyperparams(session: Session = Depends(get_db)):
    """取得所有超參數組"""
    repo = HyperparamsRepository(session)
    all_hp = repo.get_all()
    items = [_to_summary(hp) for hp in all_hp]
    return HyperparamsListResponse(items=items, total=len(items))


@router.get("/{hp_id}", response_model=HyperparamsDetail)
async def get_hyperparams(hp_id: int, session: Session = Depends(get_db)):
    """取得超參數組詳情"""
    repo = HyperparamsRepository(session)
    hp = repo.get_by_id(hp_id)
    if not hp:
        raise HTTPException(status_code=404, detail="Hyperparams not found")
    return _to_detail(hp)


@router.post("", response_model=CultivateResponse)
async def cultivate_hyperparams(
    data: CultivateRequest,
    session: Session = Depends(get_db),
):
    """
    執行超參數培養（非同步）

    基於 Walk Forward Optimization + Median Aggregation 方法
    """
    from src.repositories.database import get_session
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer

    factor_repo = FactorRepository(session)
    hp_repo = HyperparamsRepository(session)

    # 檢查名稱是否重複
    existing = hp_repo.get_by_name(data.name)
    if existing:
        raise HTTPException(status_code=400, detail="Name already exists")

    # 確認有啟用的因子
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")

    # 定義培養任務
    async def cultivation_task(progress_callback, **kwargs):
        """培養任務 wrapper"""
        task_session = get_session()
        task_factor_repo = FactorRepository(task_session)
        task_hp_repo = HyperparamsRepository(task_session)
        loop = asyncio.get_event_loop()

        try:
            # 從資料庫獲取日期範圍
            from pathlib import Path

            from sqlalchemy import func

            from src.repositories.models import StockDaily
            from src.services.qlib_exporter import ExportConfig, QlibExporter

            db_range = task_session.query(
                func.min(StockDaily.date),
                func.max(StockDaily.date),
            ).first()

            if not db_range[0] or not db_range[1]:
                raise ValueError("No stock data available")

            # 導出 qlib 資料
            await progress_callback(2, "Exporting qlib data...")

            def do_export():
                exporter = QlibExporter(task_session)
                export_config = ExportConfig(
                    start_date=db_range[0],
                    end_date=db_range[1],
                    output_dir=Path("data/qlib"),
                )
                return exporter.export(export_config)

            export_result = await asyncio.to_thread(do_export)
            await progress_callback(8, f"Exported {export_result.stocks_exported} stocks")

            factors = task_factor_repo.get_all(enabled=True)
            trainer = ModelTrainer(qlib_data_dir="data/qlib")

            # 同步回調轉非同步
            def sync_progress(progress: float, message: str):
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(progress_callback(progress, message))
                )

            # 使用 to_thread 運行同步培養
            result = await asyncio.to_thread(
                trainer.cultivate_hyperparameters,
                factors=factors,
                n_periods=data.n_periods,
                n_trials_per_period=data.n_trials_per_period,
                on_progress=sync_progress,
            )

            # 存入資料庫
            periods_data = [
                {
                    "train_start": p.train_start.isoformat(),
                    "train_end": p.train_end.isoformat(),
                    "valid_start": p.valid_start.isoformat(),
                    "valid_end": p.valid_end.isoformat(),
                    "best_ic": p.best_ic,
                    "params": p.params,
                }
                for p in result.periods
            ]

            hp = task_hp_repo.create(
                name=data.name,
                n_periods=result.n_periods,
                params=result.params,
                stability=result.stability,
                periods=periods_data,
            )

            # 廣播更新
            await broadcast_data_updated("hyperparams", "create", hp.id)

            return {
                "id": hp.id,
                "name": hp.name,
                "cultivated_at": result.cultivated_at,
                "n_periods": result.n_periods,
                "params": result.params,
                "stability": result.stability,
            }
        finally:
            task_session.close()

    # 建立非同步培養任務
    job_id = await job_manager.create_job(
        job_type="cultivate",
        task_fn=cultivation_task,
        message=f"Cultivating hyperparameters: {data.name}",
    )

    return CultivateResponse(
        job_id=job_id,
        status="queued",
        message=f"超參數培養任務已排入佇列",
    )


@router.patch("/{hp_id}", response_model=HyperparamsDetail)
async def update_hyperparams(
    hp_id: int,
    data: HyperparamsUpdate,
    session: Session = Depends(get_db),
):
    """更新名稱"""
    repo = HyperparamsRepository(session)
    hp = repo.get_by_id(hp_id)
    if not hp:
        raise HTTPException(status_code=404, detail="Hyperparams not found")

    # 檢查名稱是否重複
    existing = repo.get_by_name(data.name)
    if existing and existing.id != hp_id:
        raise HTTPException(status_code=400, detail="Name already exists")

    hp = repo.update_name(hp_id, data.name)
    await broadcast_data_updated("hyperparams", "update", hp_id)

    return _to_detail(hp)


@router.delete("/{hp_id}")
async def delete_hyperparams(hp_id: int, session: Session = Depends(get_db)):
    """刪除超參數組"""
    repo = HyperparamsRepository(session)
    hp = repo.get_by_id(hp_id)
    if not hp:
        raise HTTPException(status_code=404, detail="Hyperparams not found")

    repo.delete(hp_id)
    await broadcast_data_updated("hyperparams", "delete", hp_id)

    return {"status": "deleted", "id": hp_id}
