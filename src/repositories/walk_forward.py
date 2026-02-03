"""
Walk-Forward Backtest Repository - Walk-Forward 回測資料存取
"""

import json
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import WalkForwardBacktest


class WalkForwardBacktestRepository:
    """Walk-Forward 回測 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        start_week_id: str,
        end_week_id: str,
        initial_capital: Decimal,
        max_positions: int = 10,
        trade_price: str = "open",
        enable_incremental: bool = False,
        strategy: str = "topk",
    ) -> WalkForwardBacktest:
        """建立 Walk-Forward 回測記錄"""
        backtest = WalkForwardBacktest(
            start_week_id=start_week_id,
            end_week_id=end_week_id,
            initial_capital=initial_capital,
            max_positions=max_positions,
            trade_price=trade_price,
            enable_incremental=enable_incremental,
            strategy=strategy,
            status="queued",
        )
        self._session.add(backtest)
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def get(self, backtest_id: int) -> WalkForwardBacktest | None:
        """取得回測記錄"""
        stmt = select(WalkForwardBacktest).where(WalkForwardBacktest.id == backtest_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_recent(self, limit: int = 20) -> list[WalkForwardBacktest]:
        """取得最近的回測記錄"""
        stmt = (
            select(WalkForwardBacktest)
            .order_by(WalkForwardBacktest.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def update_status(
        self,
        backtest_id: int,
        status: str,
    ) -> WalkForwardBacktest | None:
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
        weekly_details: list[dict],
        equity_curve: list[dict],
    ) -> WalkForwardBacktest | None:
        """完成回測"""
        backtest = self.get(backtest_id)
        if not backtest:
            return None

        backtest.status = "completed"
        backtest.result = json.dumps(result)
        backtest.weekly_details = json.dumps(weekly_details)
        backtest.equity_curve = json.dumps(equity_curve)
        backtest.completed_at = datetime.now()
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def fail(self, backtest_id: int, error: str) -> WalkForwardBacktest | None:
        """標記回測失敗"""
        backtest = self.get(backtest_id)
        if not backtest:
            return None

        backtest.status = "failed"
        backtest.result = json.dumps({"error": error})
        backtest.completed_at = datetime.now()
        self._session.commit()
        self._session.refresh(backtest)
        return backtest

    def delete(self, backtest_id: int) -> bool:
        """刪除回測記錄"""
        backtest = self.get(backtest_id)
        if not backtest:
            return False

        self._session.delete(backtest)
        self._session.commit()
        return True
