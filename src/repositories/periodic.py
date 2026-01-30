"""
低頻資料 Repository 實作（月度）
"""

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repositories.models import StockMonthlyRevenue
from src.shared.types import MonthlyRevenue


class MonthlyRevenueRepository:
    """月營收 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def get(
        self,
        stock_id: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> list[MonthlyRevenue]:
        """取得指定區間的月營收"""
        # 將 year/month 轉換為可比較的數值
        start_val = start_year * 100 + start_month
        end_val = end_year * 100 + end_month

        stmt = (
            select(StockMonthlyRevenue)
            .where(
                StockMonthlyRevenue.stock_id == stock_id,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) >= start_val,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) <= end_val,
            )
            .order_by(StockMonthlyRevenue.year, StockMonthlyRevenue.month)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(r) for r in rows]

    def get_existing_months(
        self,
        stock_id: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> set[tuple[int, int]]:
        """取得已存在的 (year, month) 組合"""
        start_val = start_year * 100 + start_month
        end_val = end_year * 100 + end_month

        stmt = (
            select(StockMonthlyRevenue.year, StockMonthlyRevenue.month)
            .where(
                StockMonthlyRevenue.stock_id == stock_id,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) >= start_val,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) <= end_val,
            )
        )
        return {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

    def upsert(self, data: MonthlyRevenue) -> None:
        """新增或更新"""
        stmt = insert(StockMonthlyRevenue).values(
            stock_id=data.stock_id,
            year=data.year,
            month=data.month,
            revenue=data.revenue,
            revenue_yoy=data.revenue_yoy,
            revenue_mom=data.revenue_mom,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "year", "month"],
            set_={
                "revenue": stmt.excluded.revenue,
                "revenue_yoy": stmt.excluded.revenue_yoy,
                "revenue_mom": stmt.excluded.revenue_mom,
            },
        )
        self._session.execute(stmt)

    def upsert_many(self, items: list[MonthlyRevenue]) -> int:
        """批次新增或更新"""
        count = 0
        for item in items:
            self.upsert(item)
            count += 1
        self._session.commit()
        return count

    def _to_dataclass(self, row: StockMonthlyRevenue) -> MonthlyRevenue:
        return MonthlyRevenue(
            stock_id=row.stock_id,
            year=row.year,
            month=row.month,
            revenue=row.revenue,
            revenue_yoy=row.revenue_yoy,
            revenue_mom=row.revenue_mom,
        )
