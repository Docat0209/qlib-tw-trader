"""
訓練品質監控服務

追蹤連續週的因子 Jaccard 相似度和 IC 穩定性，
並在品質下降時產生警報。

警報閾值（來自 constants.py）：
- Jaccard 相似度 < 0.3 → low_factor_stability
- IC 標準差 > 0.1 → high_ic_volatility
- ICIR < 0.5 → low_icir
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.repositories.models import TrainingQualityMetrics, TrainingRun
from src.shared.constants import (
    QUALITY_IC_STD_MAX,
    QUALITY_ICIR_MIN,
    QUALITY_JACCARD_MIN,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """品質指標結果"""

    training_run_id: int
    factor_jaccard_sim: float | None
    factor_overlap_count: int | None
    ic_moving_avg_5w: float | None
    ic_moving_std_5w: float | None
    icir_5w: float | None
    has_warning: bool
    warning_type: str | None
    warning_message: str | None


class QualityMonitor:
    """訓練品質監控器"""

    LOOKBACK_WEEKS = 5  # 計算移動統計的週數

    def __init__(self, session: Session):
        self._session = session

    def compute_metrics(self, training_run: TrainingRun) -> QualityMetrics:
        """
        計算訓練品質指標

        Args:
            training_run: 當前訓練執行記錄

        Returns:
            QualityMetrics
        """
        # 取得最近的訓練記錄（包含當前）
        recent_runs = self._get_recent_runs(training_run, self.LOOKBACK_WEEKS)

        # 計算因子 Jaccard 相似度
        jaccard_sim, overlap_count = self._compute_factor_jaccard(
            training_run, recent_runs
        )

        # 計算 IC 移動統計
        ic_avg, ic_std, icir = self._compute_ic_stats(recent_runs)

        # 檢查警報
        warning_type, warning_message = self._check_warnings(
            jaccard_sim, ic_std, icir
        )

        return QualityMetrics(
            training_run_id=training_run.id,
            factor_jaccard_sim=jaccard_sim,
            factor_overlap_count=overlap_count,
            ic_moving_avg_5w=ic_avg,
            ic_moving_std_5w=ic_std,
            icir_5w=icir,
            has_warning=warning_type is not None,
            warning_type=warning_type,
            warning_message=warning_message,
        )

    def save_metrics(self, metrics: QualityMetrics) -> TrainingQualityMetrics:
        """
        儲存品質指標到資料庫

        Args:
            metrics: 品質指標

        Returns:
            TrainingQualityMetrics 資料庫記錄
        """
        # 檢查是否已存在
        existing = (
            self._session.query(TrainingQualityMetrics)
            .filter(TrainingQualityMetrics.training_run_id == metrics.training_run_id)
            .first()
        )

        if existing:
            # 更新現有記錄
            existing.factor_jaccard_sim = metrics.factor_jaccard_sim
            existing.factor_overlap_count = metrics.factor_overlap_count
            existing.ic_moving_avg_5w = metrics.ic_moving_avg_5w
            existing.ic_moving_std_5w = metrics.ic_moving_std_5w
            existing.icir_5w = metrics.icir_5w
            existing.has_warning = metrics.has_warning
            existing.warning_type = metrics.warning_type
            existing.warning_message = metrics.warning_message
            record = existing
        else:
            # 建立新記錄
            record = TrainingQualityMetrics(
                training_run_id=metrics.training_run_id,
                factor_jaccard_sim=metrics.factor_jaccard_sim,
                factor_overlap_count=metrics.factor_overlap_count,
                ic_moving_avg_5w=metrics.ic_moving_avg_5w,
                ic_moving_std_5w=metrics.ic_moving_std_5w,
                icir_5w=metrics.icir_5w,
                has_warning=metrics.has_warning,
                warning_type=metrics.warning_type,
                warning_message=metrics.warning_message,
            )
            self._session.add(record)

        self._session.commit()

        if metrics.has_warning:
            logger.warning(
                f"Training quality warning for run {metrics.training_run_id}: "
                f"{metrics.warning_type} - {metrics.warning_message}"
            )

        return record

    def compute_and_save(self, training_run: TrainingRun) -> TrainingQualityMetrics:
        """計算並儲存品質指標"""
        metrics = self.compute_metrics(training_run)
        return self.save_metrics(metrics)

    def get_latest_metrics(self, limit: int = 10) -> list[dict]:
        """取得最近的品質指標"""
        records = (
            self._session.query(TrainingQualityMetrics)
            .join(TrainingRun)
            .order_by(desc(TrainingRun.started_at))
            .limit(limit)
            .all()
        )

        return [
            {
                "training_run_id": r.training_run_id,
                "week_id": r.training_run.week_id if r.training_run else None,
                "factor_jaccard_sim": r.factor_jaccard_sim,
                "factor_overlap_count": r.factor_overlap_count,
                "ic_moving_avg_5w": r.ic_moving_avg_5w,
                "ic_moving_std_5w": r.ic_moving_std_5w,
                "icir_5w": r.icir_5w,
                "has_warning": r.has_warning,
                "warning_type": r.warning_type,
                "warning_message": r.warning_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]

    def _get_recent_runs(
        self, current_run: TrainingRun, n_weeks: int
    ) -> list[TrainingRun]:
        """取得最近 N 週的訓練記錄"""
        runs = (
            self._session.query(TrainingRun)
            .filter(TrainingRun.status == "completed")
            .filter(TrainingRun.model_ic.isnot(None))
            .order_by(desc(TrainingRun.started_at))
            .limit(n_weeks + 1)  # 多取一個以確保包含當前
            .all()
        )

        # 確保當前 run 在列表中
        run_ids = [r.id for r in runs]
        if current_run.id not in run_ids:
            runs = [current_run] + runs[:n_weeks]
        else:
            # 從當前 run 開始往前取
            idx = run_ids.index(current_run.id)
            runs = runs[idx : idx + n_weeks]

        return runs

    def _compute_factor_jaccard(
        self, current_run: TrainingRun, recent_runs: list[TrainingRun]
    ) -> tuple[float | None, int | None]:
        """計算因子 Jaccard 相似度（與上一週比較）"""
        if len(recent_runs) < 2:
            return None, None

        # 取得當前和上一週的選中因子
        current_factors = self._get_selected_factor_ids(current_run)
        previous_run = recent_runs[1] if recent_runs[0].id == current_run.id else recent_runs[0]
        previous_factors = self._get_selected_factor_ids(previous_run)

        if not current_factors or not previous_factors:
            return None, None

        # 計算 Jaccard 相似度
        intersection = current_factors & previous_factors
        union = current_factors | previous_factors

        jaccard = len(intersection) / len(union) if union else 0.0
        overlap_count = len(intersection)

        return jaccard, overlap_count

    def _get_selected_factor_ids(self, run: TrainingRun) -> set[int]:
        """取得訓練記錄的選中因子 ID 集合"""
        if not run.selected_factor_ids:
            return set()

        try:
            ids = json.loads(run.selected_factor_ids)
            return set(ids)
        except (json.JSONDecodeError, TypeError):
            return set()

    def _compute_ic_stats(
        self, recent_runs: list[TrainingRun]
    ) -> tuple[float | None, float | None, float | None]:
        """計算 IC 移動統計"""
        ics = [float(r.model_ic) for r in recent_runs if r.model_ic is not None]

        if len(ics) < 2:
            return None, None, None

        ic_avg = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1))  # 樣本標準差

        # ICIR = mean(IC) / std(IC)
        icir = ic_avg / ic_std if ic_std > 0 else None

        return ic_avg, ic_std, icir

    def _check_warnings(
        self,
        jaccard_sim: float | None,
        ic_std: float | None,
        icir: float | None,
    ) -> tuple[str | None, str | None]:
        """檢查是否需要產生警報"""
        warnings = []

        if jaccard_sim is not None and jaccard_sim < QUALITY_JACCARD_MIN:
            warnings.append(
                (
                    "low_factor_stability",
                    f"因子 Jaccard 相似度 {jaccard_sim:.2f} < {QUALITY_JACCARD_MIN}",
                )
            )

        if ic_std is not None and ic_std > QUALITY_IC_STD_MAX:
            warnings.append(
                (
                    "high_ic_volatility",
                    f"IC 標準差 {ic_std:.3f} > {QUALITY_IC_STD_MAX}",
                )
            )

        if icir is not None and icir < QUALITY_ICIR_MIN:
            warnings.append(
                (
                    "low_icir",
                    f"ICIR {icir:.2f} < {QUALITY_ICIR_MIN}",
                )
            )

        if not warnings:
            return None, None

        # 返回最嚴重的警報（以 Jaccard 最優先）
        priority = ["low_factor_stability", "high_ic_volatility", "low_icir"]
        for warning_type in priority:
            for wtype, wmsg in warnings:
                if wtype == warning_type:
                    return wtype, wmsg

        return warnings[0]
