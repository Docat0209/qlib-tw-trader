"""
DataService - 資料查詢統一入口

查詢邏輯：
1. 先查資料庫
2. 若資料不完整，依優先級從 Adapter 補全
3. 儲存補全的資料
4. 返回完整資料
"""

from datetime import date
from enum import Enum, auto
from typing import TypeVar

from sqlalchemy.orm import Session

from src.adapters.finmind import (
    FinMindInstitutionalAdapter,
    FinMindMarginAdapter,
    FinMindOHLCVAdapter,
    FinMindPERAdapter,
    FinMindShareholdingAdapter,
)
from src.adapters.twse import (
    TwseBulkInstitutionalAdapter,
    TwseBulkMarginAdapter,
    TwseBulkOHLCVAdapter,
    TwseBulkPERAdapter,
    TwseBulkShareholdingAdapter,
    TwseStockOHLCVAdapter,
)
from src.adapters.yfinance import YFinanceAdjCloseAdapter
from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    OHLCVRepository,
    PERRepository,
    ShareholdingRepository,
)
from src.repositories.database import get_session
from src.shared.types import (
    AdjClose,
    Institutional,
    Margin,
    OHLCV,
    PER,
    Shareholding,
)

T = TypeVar("T")


class Dataset(Enum):
    """資料集類型"""

    OHLCV = auto()
    ADJ_CLOSE = auto()
    PER = auto()
    INSTITUTIONAL = auto()
    MARGIN = auto()
    SHAREHOLDING = auto()


class DataService:
    """資料查詢統一入口"""

    def __init__(self, session: Session | None = None):
        self._session = session or get_session()
        self._init_repositories()
        self._init_adapters()

    def _init_repositories(self):
        """初始化 Repositories"""
        self._repos = {
            Dataset.OHLCV: OHLCVRepository(self._session),
            Dataset.ADJ_CLOSE: AdjCloseRepository(self._session),
            Dataset.PER: PERRepository(self._session),
            Dataset.INSTITUTIONAL: InstitutionalRepository(self._session),
            Dataset.MARGIN: MarginRepository(self._session),
            Dataset.SHAREHOLDING: ShareholdingRepository(self._session),
        }

    def _init_adapters(self):
        """初始化 Adapters（按優先級排序）"""
        self._stock_adapters = {
            Dataset.OHLCV: [TwseStockOHLCVAdapter(), FinMindOHLCVAdapter()],
            Dataset.ADJ_CLOSE: [YFinanceAdjCloseAdapter()],
            Dataset.PER: [FinMindPERAdapter()],
            Dataset.INSTITUTIONAL: [FinMindInstitutionalAdapter()],
            Dataset.MARGIN: [FinMindMarginAdapter()],
            Dataset.SHAREHOLDING: [FinMindShareholdingAdapter()],
        }
        self._bulk_adapters = {
            Dataset.OHLCV: TwseBulkOHLCVAdapter(),
            Dataset.PER: TwseBulkPERAdapter(),
            Dataset.INSTITUTIONAL: TwseBulkInstitutionalAdapter(),
            Dataset.MARGIN: TwseBulkMarginAdapter(),
            Dataset.SHAREHOLDING: TwseBulkShareholdingAdapter(),
        }

    async def get(
        self,
        dataset: Dataset,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list:
        """
        取得個股資料

        Args:
            dataset: 資料集類型
            stock_id: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            auto_fetch: 是否自動從 Adapter 補全缺失資料

        Returns:
            資料列表
        """
        repo = self._repos.get(dataset)
        if repo is None:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # 1. 先從 DB 取得現有資料
        existing = repo.get(stock_id, start_date, end_date)

        if not auto_fetch:
            return existing

        # 2. 檢查是否有缺失（簡單版：只檢查首尾）
        existing_dates = {d.date for d in existing}
        if start_date in existing_dates and end_date in existing_dates:
            return existing

        # 3. 從 Adapter 補全
        adapters = self._stock_adapters.get(dataset, [])
        for adapter in adapters:
            try:
                fetched = await adapter.fetch(stock_id, start_date, end_date)
                if fetched:
                    # 儲存到 DB
                    repo.upsert(fetched)
                    # 重新查詢（確保資料一致性）
                    return repo.get(stock_id, start_date, end_date)
            except Exception:
                continue

        return existing

    async def fetch_bulk(
        self,
        dataset: Dataset,
        target_date: date,
    ) -> list:
        """
        取得全市場當日資料（使用 Bulk Adapter）

        Args:
            dataset: 資料集類型
            target_date: 目標日期

        Returns:
            全市場資料列表
        """
        adapter = self._bulk_adapters.get(dataset)
        if adapter is None:
            raise ValueError(f"No bulk adapter for dataset: {dataset}")

        # 從 API 取得
        data = await adapter.fetch_all(target_date)

        # 儲存到 DB
        if data:
            repo = self._repos.get(dataset)
            if repo:
                repo.upsert(data)

        return data

    async def get_ohlcv(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[OHLCV]:
        """取得日K線資料（快捷方法）"""
        return await self.get(Dataset.OHLCV, stock_id, start_date, end_date, auto_fetch)

    async def get_adj_close(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[AdjClose]:
        """取得還原股價（快捷方法）"""
        return await self.get(Dataset.ADJ_CLOSE, stock_id, start_date, end_date, auto_fetch)

    async def get_per(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[PER]:
        """取得 PER/PBR（快捷方法）"""
        return await self.get(Dataset.PER, stock_id, start_date, end_date, auto_fetch)

    async def get_institutional(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[Institutional]:
        """取得三大法人（快捷方法）"""
        return await self.get(Dataset.INSTITUTIONAL, stock_id, start_date, end_date, auto_fetch)

    async def get_margin(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[Margin]:
        """取得融資融券（快捷方法）"""
        return await self.get(Dataset.MARGIN, stock_id, start_date, end_date, auto_fetch)

    async def get_shareholding(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
        auto_fetch: bool = True,
    ) -> list[Shareholding]:
        """取得外資持股（快捷方法）"""
        return await self.get(Dataset.SHAREHOLDING, stock_id, start_date, end_date, auto_fetch)
