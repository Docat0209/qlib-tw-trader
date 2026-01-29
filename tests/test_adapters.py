from datetime import date

import pytest

from src.adapters.finmind import FinMindAdapter
from src.adapters.twse import TwseAdapter


@pytest.mark.asyncio
async def test_twse_fetch_daily_ohlcv():
    """測試 TWSE 取得台積電日K線"""
    adapter = TwseAdapter()
    data = await adapter.fetch_daily_ohlcv(
        stock_id="2330",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
    )

    assert len(data) > 0
    assert data[0].stock_id == "2330"
    assert data[0].open > 0
    assert data[0].volume > 0


@pytest.mark.asyncio
async def test_finmind_fetch_daily_ohlcv():
    """測試 FinMind 取得台積電日K線（無 token 限流較嚴）"""
    adapter = FinMindAdapter()
    data = await adapter.fetch_daily_ohlcv(
        stock_id="2330",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
    )

    assert len(data) > 0
    assert data[0].stock_id == "2330"
    assert data[0].open > 0
    assert data[0].volume > 0
