from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Trade


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
