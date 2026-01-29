"""
Service 測試
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.repositories.database import Base
from src.services.data_service import DataService, Dataset


@pytest.fixture
def session():
    """建立測試用的記憶體資料庫"""
    engine = create_engine("sqlite:///:memory:")
    from src.repositories import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestDataService:
    """DataService 測試"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_with_auto_fetch(self, session):
        """測試自動補全 OHLCV"""
        ds = DataService(session)
        data = await ds.get_ohlcv(
            stock_id="2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            auto_fetch=True,
        )

        assert len(data) > 0
        assert data[0].stock_id == "2330"
        assert data[0].open > 0

    @pytest.mark.asyncio
    async def test_get_ohlcv_from_cache(self, session):
        """測試從 DB 快取讀取"""
        ds = DataService(session)

        # 第一次呼叫會從 API 取得
        data1 = await ds.get_ohlcv("2330", date(2024, 1, 2), date(2024, 1, 5))
        assert len(data1) > 0

        # 第二次呼叫應該從 DB 取得（不會再呼叫 API）
        data2 = await ds.get_ohlcv(
            "2330", date(2024, 1, 2), date(2024, 1, 5), auto_fetch=False
        )
        assert len(data2) == len(data1)

    @pytest.mark.asyncio
    async def test_get_per(self, session):
        """測試 PER"""
        ds = DataService(session)
        data = await ds.get_per("2330", date(2024, 1, 2), date(2024, 1, 5))

        assert len(data) > 0
        assert data[0].stock_id == "2330"

    @pytest.mark.asyncio
    async def test_get_adj_close(self, session):
        """測試還原股價"""
        ds = DataService(session)
        data = await ds.get_adj_close("2330", date(2024, 1, 2), date(2024, 1, 5))

        assert len(data) > 0
        assert data[0].adj_close > 0

    @pytest.mark.asyncio
    async def test_get_institutional(self, session):
        """測試三大法人"""
        ds = DataService(session)
        data = await ds.get_institutional("2330", date(2024, 1, 2), date(2024, 1, 5))

        assert len(data) > 0
        # 至少有交易
        first = data[0]
        total = first.foreign_buy + first.trust_buy + first.dealer_buy
        assert total > 0

    @pytest.mark.asyncio
    async def test_get_margin(self, session):
        """測試融資融券"""
        ds = DataService(session)
        data = await ds.get_margin("2330", date(2024, 1, 2), date(2024, 1, 5))

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_fetch_bulk_ohlcv(self, session):
        """測試 Bulk OHLCV"""
        ds = DataService(session)
        today = date.today()
        data = await ds.fetch_bulk(Dataset.OHLCV, today)

        # 可能是非交易日
        if data:
            assert len(data) > 100
