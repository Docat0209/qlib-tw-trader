"""
Dashboard API
"""

from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.dashboard import (
    DashboardSummary,
    DataStatusSummary,
    FactorsSummary,
    ModelSummary,
    PerformanceSummaryBrief,
    PredictionSummary,
)
from src.repositories.factor import FactorRepository
from src.repositories.training import TrainingRepository

router = APIRouter()

LOW_SELECTION_THRESHOLD = 0.3  # 入選率低於 30% 視為低選擇率


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    session: Session = Depends(get_db),
):
    """取得 Dashboard 摘要"""
    factor_repo = FactorRepository(session)
    training_repo = TrainingRepository(session)

    # 因子摘要
    all_factors = factor_repo.get_all()
    enabled_factors = [f for f in all_factors if f.enabled]
    low_selection_count = 0
    for f in enabled_factors:
        stats = factor_repo.get_selection_stats(f.id)
        if stats["selection_rate"] < LOW_SELECTION_THRESHOLD and stats["times_evaluated"] > 0:
            low_selection_count += 1

    # 模型摘要
    current_model = training_repo.get_current()
    training_status = training_repo.get_status()

    model_summary = ModelSummary(
        last_trained_at=training_status["last_trained_at"].isoformat() if training_status["last_trained_at"] else None,
        days_since_training=training_status["days_since_training"],
        needs_retrain=training_status["needs_retrain"],
        factor_count=current_model.factor_count if current_model else None,
        ic=float(current_model.model_ic) if current_model and current_model.model_ic else None,
        icir=float(current_model.icir) if current_model and current_model.icir else None,
    )

    # 預測摘要（待實作）
    prediction_summary = PredictionSummary(
        date=None,
        buy_signals=0,
        sell_signals=0,
        top_pick=None,
    )

    # 資料狀態摘要（簡化版）
    data_status = DataStatusSummary(
        is_complete=True,  # TODO: 從 data service 計算
        last_updated=date.today().isoformat(),
        missing_count=0,
    )

    # 績效摘要（空數據，待實作）
    performance = PerformanceSummaryBrief(
        today_return=None,
        mtd_return=None,
        ytd_return=None,
        total_return=None,
    )

    return DashboardSummary(
        factors=FactorsSummary(
            total=len(all_factors),
            enabled=len(enabled_factors),
            low_selection_count=low_selection_count,
        ),
        model=model_summary,
        prediction=prediction_summary,
        data_status=data_status,
        performance=performance,
    )
