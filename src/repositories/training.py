from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.repositories.models import TrainingFactorResult, TrainingRun

TZ_TAIPEI = ZoneInfo("Asia/Taipei")


class TrainingRepository:
    """訓練記錄存取"""

    def __init__(self, session: Session):
        self._session = session

    def create_run(
        self,
        train_start: date | None = None,
        train_end: date | None = None,
        valid_start: date | None = None,
        valid_end: date | None = None,
        name: str | None = None,
    ) -> TrainingRun:
        """建立訓練執行記錄"""
        run = TrainingRun(
            name=name,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            status="queued",
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def complete_run(
        self,
        run_id: int,
        model_ic: float,
        icir: float | None = None,
        factor_count: int | None = None,
    ) -> None:
        """完成訓練執行"""
        stmt = select(TrainingRun).where(TrainingRun.id == run_id)
        run = self._session.execute(stmt).scalar()
        if run:
            run.completed_at = datetime.now(TZ_TAIPEI)
            run.model_ic = model_ic
            run.icir = icir
            run.factor_count = factor_count
            run.status = "completed"
            self._session.commit()

    def add_factor_result(
        self,
        run_id: int,
        factor_id: int,
        ic_value: float,
        selected: bool,
    ) -> TrainingFactorResult:
        """新增因子結果"""
        result = TrainingFactorResult(
            training_run_id=run_id,
            factor_id=factor_id,
            ic_value=ic_value,
            selected=selected,
        )
        self._session.add(result)
        self._session.commit()
        self._session.refresh(result)
        return result

    def get_by_id(self, run_id: int) -> TrainingRun | None:
        """依 ID 取得訓練記錄"""
        stmt = select(TrainingRun).where(TrainingRun.id == run_id)
        return self._session.execute(stmt).scalar()

    def get_current(self) -> TrainingRun | None:
        """取得當前模型（最新完成的）"""
        return self.get_latest_run()

    def get_latest_run(self) -> TrainingRun | None:
        """取得最新的訓練記錄"""
        stmt = (
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .order_by(TrainingRun.name.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()

    def get_history(self, limit: int = 20) -> list[TrainingRun]:
        """取得歷史訓練記錄"""
        stmt = (
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .order_by(TrainingRun.name.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_selected_factors(self, run_id: int) -> list[TrainingFactorResult]:
        """取得訓練中被選中的因子"""
        stmt = (
            select(TrainingFactorResult)
            .where(TrainingFactorResult.training_run_id == run_id)
            .where(TrainingFactorResult.selected == True)
            .order_by(TrainingFactorResult.ic_value.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_status(self) -> dict:
        """取得訓練狀態"""
        from src.shared.constants import RETRAIN_THRESHOLD_DAYS

        current = self.get_current()
        running_stmt = select(TrainingRun).where(TrainingRun.status == "running")
        running = self._session.execute(running_stmt).scalar()

        days_since = None
        if current and current.completed_at:
            # 處理 naive/aware datetime 相容性
            now = datetime.now(TZ_TAIPEI)
            completed_at = current.completed_at
            if completed_at.tzinfo is None:
                # 資料庫中的 naive datetime 視為台北時間
                completed_at = completed_at.replace(tzinfo=TZ_TAIPEI)
            days_since = (now - completed_at).days

        return {
            "last_trained_at": current.completed_at if current else None,
            "days_since_training": days_since,
            "needs_retrain": days_since is not None and days_since >= RETRAIN_THRESHOLD_DAYS,
            "retrain_threshold_days": RETRAIN_THRESHOLD_DAYS,
            "current_job": running.id if running else None,
        }

    def get_all(self) -> list[TrainingRun]:
        """取得所有訓練記錄（按名稱降序，最新的在前）"""
        stmt = select(TrainingRun).order_by(TrainingRun.name.desc())
        return list(self._session.execute(stmt).scalars().all())

    def get_all_factor_results(self, run_id: int) -> list[TrainingFactorResult]:
        """取得訓練的所有因子結果（含未選中）"""
        stmt = (
            select(TrainingFactorResult)
            .where(TrainingFactorResult.training_run_id == run_id)
            .order_by(TrainingFactorResult.ic_value.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def delete(self, run_id: int) -> bool:
        """刪除訓練記錄及相關結果"""
        run = self.get_by_id(run_id)
        if not run:
            return False

        # 刪除相關的因子結果
        delete_results_stmt = (
            select(TrainingFactorResult)
            .where(TrainingFactorResult.training_run_id == run_id)
        )
        results = self._session.execute(delete_results_stmt).scalars().all()
        for result in results:
            self._session.delete(result)

        # 刪除訓練記錄
        self._session.delete(run)
        self._session.commit()
        return True
