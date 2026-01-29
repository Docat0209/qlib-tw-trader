import json
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import TrainingFactorResult, TrainingRun

TZ_TAIPEI = ZoneInfo("Asia/Taipei")


class TrainingRepository:
    """訓練記錄存取"""

    def __init__(self, session: Session):
        self._session = session

    def create_run(self, config: dict | None = None) -> TrainingRun:
        """建立訓練執行記錄"""
        run = TrainingRun(
            config=json.dumps(config) if config else None,
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def complete_run(self, run_id: int, model_ic: float) -> None:
        """完成訓練執行"""
        stmt = select(TrainingRun).where(TrainingRun.id == run_id)
        run = self._session.execute(stmt).scalar()
        if run:
            run.completed_at = datetime.now(TZ_TAIPEI)
            run.model_ic = model_ic
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

    def get_latest_run(self) -> TrainingRun | None:
        """取得最新的訓練記錄"""
        stmt = (
            select(TrainingRun)
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()

    def get_selected_factors(self, run_id: int) -> list[TrainingFactorResult]:
        """取得訓練中被選中的因子"""
        stmt = (
            select(TrainingFactorResult)
            .where(TrainingFactorResult.training_run_id == run_id)
            .where(TrainingFactorResult.selected == True)
            .order_by(TrainingFactorResult.ic_value.desc())
        )
        return list(self._session.execute(stmt).scalars().all())
