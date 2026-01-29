from abc import ABC, abstractmethod
from datetime import date

from src.shared.types import OHLCV


class DataSourceAdapter(ABC):
    """資料來源抽象介面"""

    @abstractmethod
    async def fetch_daily_ohlcv(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list[OHLCV]:
        """取得日K線資料"""
        pass
