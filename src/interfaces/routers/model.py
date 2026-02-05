"""
模型管理 API
"""

import asyncio
import json
import shutil
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.model import (
    CultivationRequest,
    CultivationResponse,
    DataRange,
    FactorSummary,
    HyperparamsInfo,
    ModelComparisonItem,
    ModelComparisonResponse,
    ModelHistoryResponse,
    ModelListResponse,
    ModelMetrics,
    ModelResponse,
    ModelStatus,
    ModelSummary,
    Period,
    QualityMetricsItem,
    QualityResponse,
    SelectionInfo,
    TrainBatchRequest,
    TrainRequest,
    TrainResponse,
    WeekModel,
    WeekSlot,
    WeeksResponse,
)
from src.repositories.factor import FactorRepository
from src.repositories.training import TrainingRepository
from src.services.job_manager import broadcast_data_updated
from src.shared.constants import EMBARGO_DAYS, RETRAIN_THRESHOLD_DAYS, TRAIN_DAYS, VALID_DAYS
from src.shared.week_utils import (
    compute_factor_pool_hash,
    compute_week_id,
    get_current_week_id,
    get_trainable_weeks,
    get_week_valid_end,
)

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
        selection_method=run.selection_method,
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

    # 解析 selection 資訊
    selection = None
    if run.selection_method:
        selection = SelectionInfo(
            method=run.selection_method,
            config=json.loads(run.selection_config) if run.selection_config else None,
            stats=json.loads(run.selection_stats) if run.selection_stats else None,
        )

    return ModelResponse(
        id=f"m{run.id:03d}",
        name=run.name,
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
        candidate_factors=candidate_factors or [],
        selected_factors=selected_factors or [],
        selection=selection,
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
    from sqlalchemy import func
    from src.repositories.models import StockDaily

    repo = TrainingRepository(session)
    status = repo.get_status()

    # 計算週訓練相關資訊
    current_week = get_current_week_id()

    # 取得最新已訓練的週
    all_runs = repo.get_all()
    trained_weeks = [r.week_id for r in all_runs if r.week_id and r.status == "completed"]
    latest_trained_week = max(trained_weeks) if trained_weeks else None

    # 計算未訓練週數
    db_range = session.query(
        func.min(StockDaily.date),
        func.max(StockDaily.date),
    ).first()

    untrained_count = 0
    current_hash = None
    if db_range and db_range[0] and db_range[1]:
        factor_repo = FactorRepository(session)
        enabled_factors = factor_repo.get_all(enabled=True)
        current_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

        trainable_weeks = get_trainable_weeks(
            data_start=db_range[0],
            data_end=db_range[1],
            session=session,
        )
        trained_set = set(trained_weeks)
        untrained_count = sum(1 for w in trainable_weeks if w.week_id not in trained_set)

    return ModelStatus(
        **status,
        current_week_id=current_week,
        latest_trained_week=latest_trained_week,
        untrained_weeks_count=untrained_count,
        current_factor_pool_hash=current_hash,
    )


@router.get("/data-range")
async def get_data_range(session: Session = Depends(get_db)):
    """取得資料庫中可用的日期範圍（用於訓練/回測）"""
    from sqlalchemy import func
    from src.repositories.models import StockDaily

    # 查詢 stock_daily 表的日期範圍
    result = session.query(
        func.min(StockDaily.date),
        func.max(StockDaily.date),
    ).first()

    if not result or not result[0] or not result[1]:
        raise HTTPException(status_code=404, detail="No stock data found in database")

    return {
        "start": result[0].isoformat(),
        "end": result[1].isoformat(),
    }


@router.get("/weeks", response_model=WeeksResponse)
async def list_weeks(session: Session = Depends(get_db)):
    """
    取得所有可訓練週的狀態

    Returns:
        slots: 週列表（最新在前）
        current_factor_pool_hash: 當前因子池 hash
        data_range: 資料庫日期範圍
    """
    from sqlalchemy import func
    from src.repositories.models import StockDaily

    # 取得資料範圍
    db_range = session.query(
        func.min(StockDaily.date),
        func.max(StockDaily.date),
    ).first()

    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=404, detail="No stock data found")

    data_start, data_end = db_range[0], db_range[1]

    # 計算當前因子池 hash
    factor_repo = FactorRepository(session)
    enabled_factors = factor_repo.get_all(enabled=True)
    current_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

    # 取得所有週（包含資料不足的）
    all_weeks = get_trainable_weeks(
        data_start=data_start,
        data_end=data_end,
        session=session,
        include_insufficient=True,
    )

    # 取得所有已訓練的模型（按 week_id 索引）
    training_repo = TrainingRepository(session)
    all_runs = training_repo.get_all()
    runs_by_week = {}
    for run in all_runs:
        if run.week_id and run.status == "completed":
            runs_by_week[run.week_id] = run

    # 建立週列表
    slots = []
    for week in all_weeks:
        run = runs_by_week.get(week.week_id)

        # 判斷狀態
        if not week.is_trainable:
            # 資料不足
            status = "insufficient_data"
            model = None
        elif run:
            # 已訓練
            is_outdated = run.factor_pool_hash != current_hash if run.factor_pool_hash else True
            model = WeekModel(
                id=f"m{run.id:03d}",
                name=run.name or "",
                model_ic=float(run.model_ic) if run.model_ic else 0.0,
                factor_count=run.factor_count or 0,
                factor_pool_hash=run.factor_pool_hash,
                is_outdated=is_outdated,
            )
            status = "trained"
        else:
            # 可訓練
            model = None
            status = "trainable"

        slots.append(
            WeekSlot(
                week_id=week.week_id,
                valid_end=week.valid_end,
                valid_start=week.valid_start,
                train_end=week.train_end,
                train_start=week.train_start,
                status=status,
                model=model,
            )
        )

    return WeeksResponse(
        slots=slots,
        current_factor_pool_hash=current_hash,
        data_range=DataRange(start=data_start, end=data_end),
    )


@router.get("/hyperparams", response_model=HyperparamsInfo)
async def get_hyperparams():
    """取得已培養的超參數資訊"""
    from src.services.model_trainer import ModelTrainer

    info = ModelTrainer.get_cultivation_info()

    if not info:
        return HyperparamsInfo(
            cultivated_at=None,
            n_periods=None,
            params=None,
            stability=None,
            periods=None,
        )

    return HyperparamsInfo(
        cultivated_at=info.get("cultivated_at"),
        n_periods=info.get("n_periods"),
        params=info.get("params"),
        stability=info.get("stability"),
        periods=info.get("periods"),
    )


@router.get("/quality", response_model=QualityResponse)
async def get_quality_metrics(
    limit: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_db),
):
    """
    取得訓練品質指標

    追蹤連續週的因子穩定性（Jaccard 相似度）和 IC 穩定性（移動平均/標準差/ICIR）。
    當指標低於閾值時會產生警報。
    """
    from src.services.stability import QualityMonitor
    from src.shared.constants import QUALITY_IC_STD_MAX, QUALITY_ICIR_MIN, QUALITY_JACCARD_MIN

    monitor = QualityMonitor(session)
    metrics = monitor.get_latest_metrics(limit=limit)

    items = [QualityMetricsItem(**m) for m in metrics]

    return QualityResponse(
        items=items,
        thresholds={
            "jaccard_min": QUALITY_JACCARD_MIN,
            "ic_std_max": QUALITY_IC_STD_MAX,
            "icir_min": QUALITY_ICIR_MIN,
        },
    )


@router.post("/cultivate-hyperparams", response_model=CultivationResponse)
async def cultivate_hyperparams(
    data: CultivationRequest,
    session: Session = Depends(get_db),
):
    """
    執行超參數培養（非同步）

    基於 Walk Forward Optimization + Median Aggregation 方法：
    - 生成多個歷史窗口
    - 每個窗口獨立運行 Optuna
    - 取各窗口最佳參數的中位數
    - 計算穩定性指標 (CV < 0.3 = 穩定)
    """
    from src.repositories.database import get_session
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer

    factor_repo = FactorRepository(session)

    # 確認有啟用的因子
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")

    # 定義培養任務
    async def cultivation_task(progress_callback, **kwargs):
        """培養任務 wrapper"""
        task_session = get_session()
        task_factor_repo = FactorRepository(task_session)
        loop = asyncio.get_event_loop()

        try:
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

            return {
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
        message=f"Cultivating hyperparameters ({data.n_periods} periods, {data.n_trials_per_period} trials/period)",
    )

    return CultivationResponse(
        job_id=job_id,
        status="queued",
        message=f"超參數培養任務已排入佇列",
    )


@router.post("/train", response_model=TrainResponse)
async def trigger_training(
    data: TrainRequest,
    session: Session = Depends(get_db),
):
    """
    觸發週模型訓練（非同步）

    接受 week_id 參數（如 "2026W05"），自動計算訓練/驗證期間
    """
    from pathlib import Path
    from datetime import timedelta

    from sqlalchemy import func

    from src.repositories.database import get_session
    from src.repositories.models import StockDaily
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    week_id = data.week_id
    factor_repo = FactorRepository(session)

    # 從資料庫取得日期範圍
    db_range = session.query(
        func.min(StockDaily.date),
        func.max(StockDaily.date),
    ).first()

    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=400, detail="No stock data found in database")

    db_start, db_end = db_range[0], db_range[1]

    # 從 week_id 計算訓練期間
    trainable_weeks = get_trainable_weeks(
        data_start=db_start,
        data_end=db_end,
        session=session,
    )

    # 找到對應的週
    week_slot = next((w for w in trainable_weeks if w.week_id == week_id), None)
    if not week_slot:
        raise HTTPException(status_code=400, detail=f"Week {week_id} is not trainable")

    train_start = week_slot.train_start
    train_end = week_slot.train_end
    valid_start = week_slot.valid_start
    valid_end = week_slot.valid_end

    # 確認有啟用的因子
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")

    # 計算因子池 hash
    factor_pool_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

    # 定義訓練任務
    async def training_task(progress_callback, **kwargs):
        """訓練任務 wrapper"""
        task_session = get_session()
        loop = asyncio.get_event_loop()

        try:
            # Step 1: 導出 qlib 資料（含因子計算緩衝）
            await progress_callback(0, "Exporting qlib data...")

            # 因子計算需要額外的歷史資料（約 60 個交易日）
            lookback_days = 90
            export_start = train_start - timedelta(days=lookback_days)
            if export_start < db_start:
                export_start = db_start

            def do_export():
                exporter = QlibExporter(task_session)
                export_config = ExportConfig(
                    start_date=export_start,
                    end_date=valid_end,
                    output_dir=Path("data/qlib"),
                )
                return exporter.export(export_config)

            export_result = await asyncio.to_thread(do_export)
            await progress_callback(5, f"Exported {export_result.stocks_exported} stocks")

            # Step 2: 執行訓練
            trainer = ModelTrainer(qlib_data_dir="data/qlib")

            def sync_progress(progress: int, message: str):
                adjusted_progress = 5 + int(progress * 0.95)
                loop.call_soon_threadsafe(
                    lambda p=adjusted_progress, m=message: asyncio.create_task(
                        progress_callback(p, m)
                    )
                )

            result = await asyncio.to_thread(
                trainer.train,
                session=task_session,
                week_id=week_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                factor_pool_hash=factor_pool_hash,
                on_progress=sync_progress,
                hyperparams_id=data.hyperparams_id,
            )

            return {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "model_ic": result.model_ic,
                "icir": result.icir,
                "selected_factor_count": len(result.selected_factor_ids),
            }
        finally:
            task_session.close()

    # 建立非同步訓練任務
    job_id = await job_manager.create_job(
        job_type="train",
        task_fn=training_task,
        message=f"Training model: {week_id}",
    )

    return TrainResponse(
        job_id=job_id,
        status="queued",
        message=f"訓練任務已排入佇列 ({week_id})",
    )


@router.post("/train-batch", response_model=TrainResponse)
async def trigger_batch_training(
    data: TrainBatchRequest,
    session: Session = Depends(get_db),
):
    """
    批量訓練一整年的模型（後端管理佇列）

    後端會依次訓練該年所有未訓練的週，不需要前端管理佇列。
    """
    from pathlib import Path
    from datetime import timedelta

    from sqlalchemy import func

    from src.repositories.database import get_session
    from src.repositories.models import StockDaily
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    year = data.year
    factor_repo = FactorRepository(session)
    training_repo = TrainingRepository(session)

    # 從資料庫取得日期範圍
    db_range = session.query(
        func.min(StockDaily.date),
        func.max(StockDaily.date),
    ).first()

    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=400, detail="No stock data found in database")

    db_start, db_end = db_range[0], db_range[1]

    # 取得該年所有可訓練的週
    trainable_weeks = get_trainable_weeks(
        data_start=db_start,
        data_end=db_end,
        session=session,
    )

    # 過濾出該年的週
    year_weeks = [w for w in trainable_weeks if w.week_id.startswith(year)]

    # 取得已訓練的週
    all_runs = training_repo.get_all()
    trained_week_ids = {r.week_id for r in all_runs if r.week_id and r.status == "completed"}

    # 過濾出未訓練的週
    untrained_weeks = [w for w in year_weeks if w.week_id not in trained_week_ids]

    if not untrained_weeks:
        raise HTTPException(status_code=400, detail=f"{year} 年沒有待訓練的週")

    # 按週 ID 排序（從最早的週開始）
    untrained_weeks.sort(key=lambda w: w.week_id)
    week_ids = [w.week_id for w in untrained_weeks]

    # 確認有啟用的因子
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")

    factor_pool_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

    # 定義批量訓練任務
    async def batch_training_task(progress_callback, **kwargs):
        """批量訓練任務"""
        task_session = get_session()
        loop = asyncio.get_event_loop()

        try:
            total_weeks = len(week_ids)
            results = []

            for idx, week_id in enumerate(week_ids):
                # 計算整體進度
                base_progress = (idx / total_weeks) * 100
                week_progress_range = 100 / total_weeks

                await progress_callback(
                    base_progress,
                    f"[{idx + 1}/{total_weeks}] Training {week_id}..."
                )

                # 取得該週的訓練參數
                week_slot = next((w for w in untrained_weeks if w.week_id == week_id), None)
                if not week_slot:
                    continue

                train_start = week_slot.train_start
                train_end = week_slot.train_end
                valid_start = week_slot.valid_start
                valid_end = week_slot.valid_end

                # 導出 qlib 資料
                lookback_days = 90
                export_start = train_start - timedelta(days=lookback_days)
                if export_start < db_start:
                    export_start = db_start

                def do_export():
                    exporter = QlibExporter(task_session)
                    export_config = ExportConfig(
                        start_date=export_start,
                        end_date=valid_end,
                        output_dir=Path("data/qlib"),
                    )
                    return exporter.export(export_config)

                await asyncio.to_thread(do_export)

                # 執行訓練
                trainer = ModelTrainer(qlib_data_dir="data/qlib")

                def sync_progress(progress: float, message: str):
                    # 將單週進度映射到整體進度
                    adjusted = base_progress + (progress / 100) * week_progress_range
                    loop.call_soon_threadsafe(
                        lambda p=adjusted, m=f"[{idx + 1}/{total_weeks}] {message}": asyncio.create_task(
                            progress_callback(p, m)
                        )
                    )

                result = await asyncio.to_thread(
                    trainer.train,
                    session=task_session,
                    week_id=week_id,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    factor_pool_hash=factor_pool_hash,
                    on_progress=sync_progress,
                    hyperparams_id=data.hyperparams_id,
                )

                results.append({
                    "week_id": week_id,
                    "model_name": result.model_name,
                    "model_ic": result.model_ic,
                })

            return {
                "total_trained": len(results),
                "results": results,
            }
        finally:
            task_session.close()

    # 建立批量訓練任務
    job_id = await job_manager.create_job(
        job_type="train_batch",
        task_fn=batch_training_task,
        message=f"Batch training {year}: {len(week_ids)} weeks",
    )

    return TrainResponse(
        job_id=job_id,
        status="queued",
        message=f"批量訓練已排入佇列 ({year} 年 {len(week_ids)} 週)",
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
        # 先建立候選因子列表
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

        # 按 selected_ids 的順序建立 selected_factors（順序對模型預測至關重要）
        for fid in selected_ids:
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


@router.delete("/all")
async def delete_all_models(
    session: Session = Depends(get_db),
):
    """刪除所有模型"""
    repo = TrainingRepository(session)

    # 取得所有模型
    runs = repo.get_all()
    deleted_count = 0

    for run in runs:
        # 刪除模型檔案目錄
        if run.name:
            model_dir = MODELS_DIR / run.name
            if model_dir.exists():
                shutil.rmtree(model_dir)

        # 刪除資料庫記錄
        repo.delete(run.id)
        deleted_count += 1

    await broadcast_data_updated("models", "delete", "all")

    return {"deleted_count": deleted_count}


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    session: Session = Depends(get_db),
):
    """刪除模型"""
    from src.services.job_manager import job_manager

    run_id = _parse_model_id(model_id)
    repo = TrainingRepository(session)

    run = repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")

    # 如果是 running/queued，嘗試取消關聯的 job
    if run.status in ("running", "queued"):
        # 查找關聯的 job 並取消
        from src.repositories.job import JobRepository
        job_repo = JobRepository(session)
        active_jobs = job_repo.get_active()
        for job in active_jobs:
            if job.job_type == "train":
                await job_manager.cancel_job(job.id)
                session.delete(job)
        session.commit()

    # 刪除模型檔案目錄
    if run.name:
        model_dir = MODELS_DIR / run.name
        if model_dir.exists():
            shutil.rmtree(model_dir)

    # 刪除資料庫記錄
    repo.delete(run_id)

    await broadcast_data_updated("models", "delete", model_id)

    return {"status": "deleted", "id": model_id}


