"""
低頻資料 Repository 實作（月度/季度）
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repositories.models import (
    StockMonthlyRevenue,
    StockQuarterlyBalance,
    StockQuarterlyCashFlow,
    StockQuarterlyFinancial,
    StockDividend,
)
from src.shared.types import (
    Dividend,
    MonthlyRevenue,
    QuarterlyBalance,
    QuarterlyCashFlow,
    QuarterlyFinancial,
)


# =============================================================================
# 月度 Repository
# =============================================================================


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


# =============================================================================
# 季度 Repository
# =============================================================================


class QuarterlyFinancialRepository:
    """季度財報（綜合損益表）Repository"""

    def __init__(self, session: Session):
        self._session = session

    def get(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> list[QuarterlyFinancial]:
        """取得指定區間的季度財報"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyFinancial)
            .where(
                StockQuarterlyFinancial.stock_id == stock_id,
                (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) >= start_val,
                (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) <= end_val,
            )
            .order_by(StockQuarterlyFinancial.year, StockQuarterlyFinancial.quarter)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(r) for r in rows]

    def get_existing_quarters(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> set[tuple[int, int]]:
        """取得已存在的 (year, quarter) 組合"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyFinancial.year, StockQuarterlyFinancial.quarter)
            .where(
                StockQuarterlyFinancial.stock_id == stock_id,
                (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) >= start_val,
                (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) <= end_val,
            )
        )
        return {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

    def upsert(self, data: QuarterlyFinancial) -> None:
        """新增或更新"""
        stmt = insert(StockQuarterlyFinancial).values(
            stock_id=data.stock_id,
            year=data.year,
            quarter=data.quarter,
            revenue=data.revenue,
            gross_profit=data.gross_profit,
            operating_income=data.operating_income,
            net_income=data.net_income,
            eps=data.eps,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "year", "quarter"],
            set_={
                "revenue": stmt.excluded.revenue,
                "gross_profit": stmt.excluded.gross_profit,
                "operating_income": stmt.excluded.operating_income,
                "net_income": stmt.excluded.net_income,
                "eps": stmt.excluded.eps,
            },
        )
        self._session.execute(stmt)

    def upsert_many(self, items: list[QuarterlyFinancial]) -> int:
        """批次新增或更新"""
        count = 0
        for item in items:
            self.upsert(item)
            count += 1
        self._session.commit()
        return count

    def _to_dataclass(self, row: StockQuarterlyFinancial) -> QuarterlyFinancial:
        return QuarterlyFinancial(
            stock_id=row.stock_id,
            year=row.year,
            quarter=row.quarter,
            revenue=row.revenue,
            gross_profit=row.gross_profit,
            operating_income=row.operating_income,
            net_income=row.net_income,
            eps=row.eps,
        )


class QuarterlyBalanceRepository:
    """資產負債表 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def get(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> list[QuarterlyBalance]:
        """取得指定區間的資產負債表"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyBalance)
            .where(
                StockQuarterlyBalance.stock_id == stock_id,
                (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) >= start_val,
                (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) <= end_val,
            )
            .order_by(StockQuarterlyBalance.year, StockQuarterlyBalance.quarter)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(r) for r in rows]

    def get_existing_quarters(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> set[tuple[int, int]]:
        """取得已存在的 (year, quarter) 組合"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyBalance.year, StockQuarterlyBalance.quarter)
            .where(
                StockQuarterlyBalance.stock_id == stock_id,
                (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) >= start_val,
                (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) <= end_val,
            )
        )
        return {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

    def upsert(self, data: QuarterlyBalance) -> None:
        """新增或更新"""
        stmt = insert(StockQuarterlyBalance).values(
            stock_id=data.stock_id,
            year=data.year,
            quarter=data.quarter,
            total_assets=data.total_assets,
            total_liabilities=data.total_liabilities,
            total_equity=data.total_equity,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "year", "quarter"],
            set_={
                "total_assets": stmt.excluded.total_assets,
                "total_liabilities": stmt.excluded.total_liabilities,
                "total_equity": stmt.excluded.total_equity,
            },
        )
        self._session.execute(stmt)

    def upsert_many(self, items: list[QuarterlyBalance]) -> int:
        """批次新增或更新"""
        count = 0
        for item in items:
            self.upsert(item)
            count += 1
        self._session.commit()
        return count

    def _to_dataclass(self, row: StockQuarterlyBalance) -> QuarterlyBalance:
        return QuarterlyBalance(
            stock_id=row.stock_id,
            year=row.year,
            quarter=row.quarter,
            total_assets=row.total_assets,
            total_liabilities=row.total_liabilities,
            total_equity=row.total_equity,
        )


class QuarterlyCashFlowRepository:
    """現金流量表 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def get(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> list[QuarterlyCashFlow]:
        """取得指定區間的現金流量表"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyCashFlow)
            .where(
                StockQuarterlyCashFlow.stock_id == stock_id,
                (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) >= start_val,
                (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) <= end_val,
            )
            .order_by(StockQuarterlyCashFlow.year, StockQuarterlyCashFlow.quarter)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(r) for r in rows]

    def get_existing_quarters(
        self,
        stock_id: str,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> set[tuple[int, int]]:
        """取得已存在的 (year, quarter) 組合"""
        start_val = start_year * 10 + start_quarter
        end_val = end_year * 10 + end_quarter

        stmt = (
            select(StockQuarterlyCashFlow.year, StockQuarterlyCashFlow.quarter)
            .where(
                StockQuarterlyCashFlow.stock_id == stock_id,
                (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) >= start_val,
                (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) <= end_val,
            )
        )
        return {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

    def upsert(self, data: QuarterlyCashFlow) -> None:
        """新增或更新"""
        stmt = insert(StockQuarterlyCashFlow).values(
            stock_id=data.stock_id,
            year=data.year,
            quarter=data.quarter,
            operating_cf=data.operating_cf,
            investing_cf=data.investing_cf,
            financing_cf=data.financing_cf,
            free_cf=data.free_cf,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "year", "quarter"],
            set_={
                "operating_cf": stmt.excluded.operating_cf,
                "investing_cf": stmt.excluded.investing_cf,
                "financing_cf": stmt.excluded.financing_cf,
                "free_cf": stmt.excluded.free_cf,
            },
        )
        self._session.execute(stmt)

    def upsert_many(self, items: list[QuarterlyCashFlow]) -> int:
        """批次新增或更新"""
        count = 0
        for item in items:
            self.upsert(item)
            count += 1
        self._session.commit()
        return count

    def _to_dataclass(self, row: StockQuarterlyCashFlow) -> QuarterlyCashFlow:
        return QuarterlyCashFlow(
            stock_id=row.stock_id,
            year=row.year,
            quarter=row.quarter,
            operating_cf=row.operating_cf,
            investing_cf=row.investing_cf,
            financing_cf=row.financing_cf,
            free_cf=row.free_cf,
        )


# =============================================================================
# 事件型 Repository
# =============================================================================


class DividendRepository:
    """股利政策 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def get(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[Dividend]:
        """取得指定區間的股利資料"""
        stmt = (
            select(StockDividend)
            .where(
                StockDividend.stock_id == stock_id,
                StockDividend.ex_date >= start_date,
                StockDividend.ex_date <= end_date,
            )
            .order_by(StockDividend.ex_date)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_dataclass(r) for r in rows]

    def get_existing_dates(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        """取得已存在的除息日"""
        stmt = (
            select(StockDividend.ex_date)
            .where(
                StockDividend.stock_id == stock_id,
                StockDividend.ex_date >= start_date,
                StockDividend.ex_date <= end_date,
            )
        )
        return {r[0] for r in self._session.execute(stmt).fetchall()}

    def upsert(self, data: Dividend) -> None:
        """新增或更新"""
        stmt = insert(StockDividend).values(
            stock_id=data.stock_id,
            ex_date=data.ex_date,
            cash_dividend=data.cash_dividend,
            stock_dividend=data.stock_dividend,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "ex_date"],
            set_={
                "cash_dividend": stmt.excluded.cash_dividend,
                "stock_dividend": stmt.excluded.stock_dividend,
            },
        )
        self._session.execute(stmt)

    def upsert_many(self, items: list[Dividend]) -> int:
        """批次新增或更新"""
        count = 0
        for item in items:
            self.upsert(item)
            count += 1
        self._session.commit()
        return count

    def _to_dataclass(self, row: StockDividend) -> Dividend:
        return Dividend(
            stock_id=row.stock_id,
            ex_date=row.ex_date,
            cash_dividend=row.cash_dividend,
            stock_dividend=row.stock_dividend,
        )
