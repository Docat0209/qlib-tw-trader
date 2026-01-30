"""
Backtest Repository - 回測資料存取
"""

import json
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Backtest


class BacktestRepository:
    """回測 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        model_id: int,
        start_date: date,
        end_date: date,
        initial_capital: Decimal,
        max_positions: int = 10,
    ) -> Backtest:
        """建立回測記錄"""
        backtest = Backtest(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_positions=max_positions,
            status="queued",
        )
        self._session.add(backtest)
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def get(self, backtest_id: int) -> Backtest | None:
        """取得回測記錄"""
        stmt = select(Backtest).where(Backtest.id == backtest_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_model(self, model_id: int, limit: int = 20) -> list[Backtest]:
        """取得模型的回測記錄"""
        stmt = (
            select(Backtest)
            .where(Backtest.model_id == model_id)
            .order_by(Backtest.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_recent(self, limit: int = 20) -> list[Backtest]:
        """取得最近的回測記錄"""
        stmt = select(Backtest).order_by(Backtest.created_at.desc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def update_status(
        self,
        backtest_id: int,
        status: str,
    ) -> Backtest | None:
        """更新回測狀態"""
        backtest = self.get(backtest_id)
        if not backtest:
            return None

        backtest.status = status
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def complete(
        self,
        backtest_id: int,
        result: dict,
        equity_curve: list[dict],
    ) -> Backtest | None:
        """完成回測"""
        backtest = self.get(backtest_id)
        if not backtest:
            return None

        backtest.status = "completed"
        backtest.result = json.dumps(result)
        backtest.equity_curve = json.dumps(equity_curve)
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def fail(self, backtest_id: int, error: str) -> Backtest | None:
        """標記回測失敗"""
        backtest = self.get(backtest_id)
        if not backtest:
            return None

        backtest.status = "failed"
        backtest.result = json.dumps({"error": error})
        self._session.commit()
        self._session.refresh(backtest)
        return backtest
