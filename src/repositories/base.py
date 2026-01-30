"""
Repository 基礎類別
"""

from datetime import date
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repositories.database import Base

T = TypeVar("T")  # Dataclass type
M = TypeVar("M", bound=Base)  # Model type


class BaseRepository(Generic[T, M]):
    """泛型 Repository 基礎類別"""

    def __init__(self, session: Session, model: type[M]):
        self._session = session
        self._model = model

    def _to_dataclass(self, row: M) -> T:
        """將 Model 轉換為 Dataclass（子類別實作）"""
        raise NotImplementedError

    def _to_dict(self, data: T) -> dict:
        """將 Dataclass 轉換為 dict（子類別實作）"""
        raise NotImplementedError

    def _get_conflict_keys(self) -> list[str]:
        """取得 upsert 衝突鍵（子類別實作）"""
        raise NotImplementedError

    def _get_update_fields(self) -> list[str]:
        """取得 upsert 更新欄位（子類別實作）"""
        raise NotImplementedError

    def upsert(self, data: list[T]) -> int:
        """批次新增或更新，回傳影響筆數"""
        if not data:
            return 0

        values = [self._to_dict(d) for d in data]
        stmt = insert(self._model).values(values)

        update_dict = {field: getattr(stmt.excluded, field) for field in self._get_update_fields()}
        stmt = stmt.on_conflict_do_update(
            index_elements=self._get_conflict_keys(),
            set_=update_dict,
        )

        result = self._session.execute(stmt)
        self._session.commit()
        return result.rowcount


class StockDailyRepository(BaseRepository[T, M]):
    """個股日頻資料 Repository（stock_id + date 為主鍵）"""

    def get(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[T]:
        """取得指定區間資料"""
        stmt = (
            select(self._model)
            .where(self._model.stock_id == stock_id)
            .where(self._model.date >= start_date)
            .where(self._model.date <= end_date)
            .order_by(self._model.date)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(row) for row in rows]

    def get_latest_date(self, stock_id: str) -> date | None:
        """取得某股票最新資料日期"""
        stmt = (
            select(self._model.date)
            .where(self._model.stock_id == stock_id)
            .order_by(self._model.date.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()

    def get_missing_dates(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        trading_dates: list[date],
    ) -> list[date]:
        """取得缺失的日期"""
        stmt = (
            select(self._model.date)
            .where(self._model.stock_id == stock_id)
            .where(self._model.date >= start_date)
            .where(self._model.date <= end_date)
        )
        existing = set(self._session.execute(stmt).scalars().all())
        return [d for d in trading_dates if start_date <= d <= end_date and d not in existing]

    def count(self, stock_id: str) -> int:
        """取得某股票總資料筆數"""
        from sqlalchemy import func

        stmt = select(func.count()).where(self._model.stock_id == stock_id)
        return self._session.execute(stmt).scalar() or 0

    def get_earliest_date(self, stock_id: str) -> date | None:
        """取得某股票最早資料日期"""
        stmt = (
            select(self._model.date)
            .where(self._model.stock_id == stock_id)
            .order_by(self._model.date.asc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()

    def get_all_stock_ids(self) -> list[str]:
        """取得所有有資料的股票代碼"""
        stmt = select(self._model.stock_id).distinct().order_by(self._model.stock_id)
        return list(self._session.execute(stmt).scalars().all())

    def get_global_latest_date(self) -> date | None:
        """取得整體最新資料日期（所有股票中最新的）"""
        from sqlalchemy import func

        stmt = select(func.max(self._model.date))
        return self._session.execute(stmt).scalar()

    def get_global_earliest_date(self) -> date | None:
        """取得整體最早資料日期（所有股票中最早的）"""
        from sqlalchemy import func

        stmt = select(func.min(self._model.date))
        return self._session.execute(stmt).scalar()


class MarketDailyRepository(BaseRepository[T, M]):
    """市場日頻資料 Repository（date 為主鍵）"""

    def get(
        self,
        start_date: date,
        end_date: date,
    ) -> list[T]:
        """取得指定區間資料"""
        stmt = (
            select(self._model)
            .where(self._model.date >= start_date)
            .where(self._model.date <= end_date)
            .order_by(self._model.date)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(row) for row in rows]

    def get_latest_date(self) -> date | None:
        """取得最新資料日期"""
        stmt = (
            select(self._model.date)
            .order_by(self._model.date.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()

    def get_earliest_date(self) -> date | None:
        """取得最早資料日期"""
        stmt = (
            select(self._model.date)
            .order_by(self._model.date.asc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar()
