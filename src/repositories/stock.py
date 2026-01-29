from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repositories.models import StockDaily
from src.shared.types import OHLCV


class StockRepository:
    """股票資料存取"""

    def __init__(self, session: Session):
        self._session = session

    def upsert_daily(self, data: list[OHLCV]) -> int:
        """批次新增或更新日K線資料，回傳影響筆數"""
        if not data:
            return 0

        stmt = insert(StockDaily).values(
            [
                {
                    "stock_id": d.stock_id,
                    "date": d.date,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        result = self._session.execute(stmt)
        self._session.commit()
        return result.rowcount

    def get_daily(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[OHLCV]:
        """取得日K線資料"""
        stmt = (
            select(StockDaily)
            .where(StockDaily.stock_id == stock_id)
            .where(StockDaily.date >= start_date)
            .where(StockDaily.date <= end_date)
            .order_by(StockDaily.date)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [
            OHLCV(
                date=row.date,
                stock_id=row.stock_id,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in rows
        ]

    def get_latest_date(self, stock_id: str) -> date | None:
        """取得某股票最新資料日期"""
        stmt = (
            select(StockDaily.date)
            .where(StockDaily.stock_id == stock_id)
            .order_by(StockDaily.date.desc())
            .limit(1)
        )
        result = self._session.execute(stmt).scalar()
        return result
