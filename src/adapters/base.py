from abc import ABC, abstractmethod
from datetime import date
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseAdapter(ABC, Generic[T]):
    """資料來源抽象介面"""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """來源名稱（twse/finmind/yfinance）"""
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """對應的 FinMind Dataset 名稱"""
        pass


class StockDataAdapter(BaseAdapter[T]):
    """個股資料來源（需要 stock_id）"""

    @abstractmethod
    async def fetch(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[T]:
        """取得單一股票的歷史資料"""
        pass


class MarketDataAdapter(BaseAdapter[T]):
    """市場資料來源（不需要 stock_id）"""

    @abstractmethod
    async def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> list[T]:
        """取得市場級別資料"""
        pass


class BulkDataAdapter(BaseAdapter[T]):
    """批量資料來源（一次取全市場當日資料）"""

    @abstractmethod
    async def fetch_all(
        self,
        target_date: date,
    ) -> list[T]:
        """取得單日全市場資料"""
        pass
