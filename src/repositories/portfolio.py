from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Position, Prediction, Trade


class PositionRepository:
    """持倉記錄存取"""

    def __init__(self, session: Session):
        self._session = session

    def get_all(self) -> list[Position]:
        """取得所有持倉"""
        stmt = select(Position).where(Position.shares > 0)
        return list(self._session.execute(stmt).scalars().all())

    def get_by_stock(self, stock_id: str) -> Position | None:
        """依股票代碼取得持倉"""
        stmt = select(Position).where(Position.stock_id == stock_id)
        return self._session.execute(stmt).scalar()

    def upsert(
        self,
        stock_id: str,
        shares: int,
        avg_cost: Decimal,
    ) -> Position:
        """新增或更新持倉"""
        position = self.get_by_stock(stock_id)
        if position:
            position.shares = shares
            position.avg_cost = avg_cost
        else:
            position = Position(
                stock_id=stock_id,
                shares=shares,
                avg_cost=avg_cost,
            )
            self._session.add(position)
        self._session.commit()
        self._session.refresh(position)
        return position


class TradeRepository:
    """交易記錄存取"""

    def __init__(self, session: Session):
        self._session = session

    def get_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        stock_id: str | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """取得交易記錄"""
        stmt = select(Trade).order_by(Trade.date.desc())
        if start_date:
            stmt = stmt.where(Trade.date >= start_date)
        if end_date:
            stmt = stmt.where(Trade.date <= end_date)
        if stock_id:
            stmt = stmt.where(Trade.stock_id == stock_id)
        stmt = stmt.limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def create(
        self,
        date: date,
        stock_id: str,
        side: str,
        shares: int,
        price: Decimal,
        commission: Decimal = Decimal(0),
        reason: str | None = None,
    ) -> Trade:
        """新增交易記錄"""
        trade = Trade(
            date=date,
            stock_id=stock_id,
            side=side,
            shares=shares,
            price=price,
            amount=price * shares,
            commission=commission,
            reason=reason,
        )
        self._session.add(trade)
        self._session.commit()
        self._session.refresh(trade)
        return trade


class PredictionRepository:
    """預測記錄存取"""

    def __init__(self, session: Session):
        self._session = session

    def get_latest(self) -> list[Prediction]:
        """取得最新一天的預測"""
        # 先找出最新日期
        latest_date_stmt = (
            select(Prediction.date)
            .order_by(Prediction.date.desc())
            .limit(1)
        )
        latest_date = self._session.execute(latest_date_stmt).scalar()

        if not latest_date:
            return []

        stmt = (
            select(Prediction)
            .where(Prediction.date == latest_date)
            .order_by(Prediction.rank)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_by_date(self, target_date: date) -> list[Prediction]:
        """取得指定日期的預測"""
        stmt = (
            select(Prediction)
            .where(Prediction.date == target_date)
            .order_by(Prediction.rank)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        stock_id: str | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        """取得歷史預測記錄"""
        stmt = select(Prediction).order_by(Prediction.date.desc(), Prediction.rank)
        if start_date:
            stmt = stmt.where(Prediction.date >= start_date)
        if end_date:
            stmt = stmt.where(Prediction.date <= end_date)
        if stock_id:
            stmt = stmt.where(Prediction.stock_id == stock_id)
        stmt = stmt.limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def create(
        self,
        date: date,
        model_id: int,
        stock_id: str,
        score: float,
        rank: int,
        signal: str,
    ) -> Prediction:
        """新增預測記錄"""
        prediction = Prediction(
            date=date,
            model_id=model_id,
            stock_id=stock_id,
            score=score,
            rank=rank,
            signal=signal,
        )
        self._session.add(prediction)
        self._session.commit()
        self._session.refresh(prediction)
        return prediction
