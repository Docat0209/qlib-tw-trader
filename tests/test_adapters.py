"""
Adapter 測試
"""

from datetime import date
from decimal import Decimal

import pytest

from src.adapters.finmind import (
    FinMindInstitutionalAdapter,
    FinMindMarginAdapter,
    FinMindOHLCVAdapter,
    FinMindPERAdapter,
)
from src.adapters.twse import (
    TwseBulkOHLCVAdapter,
    TwseBulkPERAdapter,
    TwseStockOHLCVAdapter,
)
from src.adapters.yfinance import YFinanceAdjCloseAdapter


class TestTwseAdapters:
    """TWSE Adapter 測試"""

    @pytest.mark.asyncio
    async def test_stock_ohlcv_adapter(self):
        """測試單一股票 OHLCV（舊版 API）"""
        adapter = TwseStockOHLCVAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        assert data[0].open > 0
        assert data[0].volume > 0

    @pytest.mark.asyncio
    async def test_bulk_ohlcv_adapter(self):
        """測試全市場 OHLCV（RWD API）"""
        adapter = TwseBulkOHLCVAdapter()
        # 使用今天的日期（API 只返回當日資料）
        today = date.today()
        data = await adapter.fetch_all(today)

        # 可能是非交易日，所以不檢查必須有資料
        if data:
            assert len(data) > 100  # 至少有 100 支股票
            tsmc = [d for d in data if d.stock_id == "2330"]
            if tsmc:
                assert tsmc[0].open > 0

    @pytest.mark.asyncio
    async def test_bulk_per_adapter(self):
        """測試全市場 PER（RWD API）"""
        adapter = TwseBulkPERAdapter()
        today = date.today()
        data = await adapter.fetch_all(today)

        if data:
            assert len(data) > 100
            tsmc = [d for d in data if d.stock_id == "2330"]
            if tsmc:
                assert tsmc[0].pe_ratio is not None or tsmc[0].pb_ratio is not None


class TestFinMindAdapters:
    """FinMind Adapter 測試"""

    @pytest.mark.asyncio
    async def test_ohlcv_adapter(self):
        """測試 OHLCV"""
        adapter = FinMindOHLCVAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        assert data[0].open > 0
        assert data[0].volume > 0

    @pytest.mark.asyncio
    async def test_per_adapter(self):
        """測試 PER/PBR"""
        adapter = FinMindPERAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        # PE/PB 可能為 None（如虧損時）
        assert data[0].pe_ratio is not None or data[0].pb_ratio is not None

    @pytest.mark.asyncio
    async def test_institutional_adapter(self):
        """測試三大法人（聚合後）"""
        adapter = FinMindInstitutionalAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        # 至少有一個法人有交易
        first = data[0]
        total_buy = first.foreign_buy + first.trust_buy + first.dealer_buy
        assert total_buy > 0

    @pytest.mark.asyncio
    async def test_margin_adapter(self):
        """測試融資融券"""
        adapter = FinMindMarginAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"


class TestYFinanceAdapter:
    """yfinance Adapter 測試"""

    @pytest.mark.asyncio
    async def test_adj_close_adapter(self):
        """測試還原股價"""
        adapter = YFinanceAdjCloseAdapter()
        data = await adapter.fetch(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        assert data[0].adj_close > 0
