"""
Repository 測試
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.repositories.database import Base
from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    OHLCVRepository,
    PERRepository,
    ShareholdingRepository,
)
from src.repositories.factor import FactorRepository
from src.repositories.stock import StockRepository
from src.repositories.training import TrainingRepository
from src.shared.types import (
    AdjClose,
    Institutional,
    Margin,
    OHLCV,
    PER,
    Shareholding,
)


@pytest.fixture
def session():
    """建立測試用的記憶體資料庫"""
    engine = create_engine("sqlite:///:memory:")
    # 導入 models 確保表格被註冊
    from src.repositories import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestOHLCVRepository:
    """OHLCV Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = OHLCVRepository(session)
        data = [
            OHLCV(
                date=date(2024, 1, 2),
                stock_id="2330",
                open=Decimal("580.00"),
                high=Decimal("585.00"),
                low=Decimal("578.00"),
                close=Decimal("583.00"),
                volume=10000000,
            ),
            OHLCV(
                date=date(2024, 1, 3),
                stock_id="2330",
                open=Decimal("583.00"),
                high=Decimal("590.00"),
                low=Decimal("582.00"),
                close=Decimal("588.00"),
                volume=12000000,
            ),
        ]

        count = repo.upsert(data)
        assert count == 2

        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))
        assert len(result) == 2
        assert result[0].close == Decimal("583.00")

    def test_get_latest_date(self, session):
        repo = OHLCVRepository(session)
        data = [
            OHLCV(
                date=date(2024, 1, 2),
                stock_id="2330",
                open=Decimal("580.00"),
                high=Decimal("585.00"),
                low=Decimal("578.00"),
                close=Decimal("583.00"),
                volume=10000000,
            ),
        ]
        repo.upsert(data)

        latest = repo.get_latest_date("2330")
        assert latest == date(2024, 1, 2)
        assert repo.get_latest_date("9999") is None


class TestAdjCloseRepository:
    """AdjClose Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = AdjCloseRepository(session)
        data = [
            AdjClose(date=date(2024, 1, 2), stock_id="2330", adj_close=Decimal("580.00")),
            AdjClose(date=date(2024, 1, 3), stock_id="2330", adj_close=Decimal("585.00")),
        ]

        repo.upsert(data)
        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))

        assert len(result) == 2
        assert result[0].adj_close == Decimal("580.00")


class TestPERRepository:
    """PER Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = PERRepository(session)
        data = [
            PER(
                date=date(2024, 1, 2),
                stock_id="2330",
                pe_ratio=Decimal("25.50"),
                pb_ratio=Decimal("5.20"),
                dividend_yield=Decimal("2.50"),
            ),
        ]

        repo.upsert(data)
        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))

        assert len(result) == 1
        assert result[0].pe_ratio == Decimal("25.50")


class TestInstitutionalRepository:
    """Institutional Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = InstitutionalRepository(session)
        data = [
            Institutional(
                date=date(2024, 1, 2),
                stock_id="2330",
                foreign_buy=1000000,
                foreign_sell=500000,
                trust_buy=200000,
                trust_sell=100000,
                dealer_buy=50000,
                dealer_sell=30000,
            ),
        ]

        repo.upsert(data)
        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))

        assert len(result) == 1
        assert result[0].foreign_buy == 1000000
        assert result[0].foreign_net == 500000


class TestMarginRepository:
    """Margin Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = MarginRepository(session)
        data = [
            Margin(
                date=date(2024, 1, 2),
                stock_id="2330",
                margin_buy=1000,
                margin_sell=500,
                margin_balance=10000,
                short_buy=100,
                short_sell=50,
                short_balance=500,
            ),
        ]

        repo.upsert(data)
        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))

        assert len(result) == 1
        assert result[0].margin_balance == 10000


class TestShareholdingRepository:
    """Shareholding Repository 測試"""

    def test_upsert_and_get(self, session):
        repo = ShareholdingRepository(session)
        data = [
            Shareholding(
                date=date(2024, 1, 2),
                stock_id="2330",
                foreign_shares=5000000000,
                foreign_ratio=Decimal("70.50"),
            ),
        ]

        repo.upsert(data)
        result = repo.get("2330", date(2024, 1, 1), date(2024, 1, 5))

        assert len(result) == 1
        assert result[0].foreign_ratio == Decimal("70.50")


class TestStockRepository:
    """舊版 StockRepository 測試（向後相容）"""

    def test_upsert_and_get_daily(self, session):
        repo = StockRepository(session)
        data = [
            OHLCV(
                date=date(2024, 1, 2),
                stock_id="2330",
                open=Decimal("580.00"),
                high=Decimal("585.00"),
                low=Decimal("578.00"),
                close=Decimal("583.00"),
                volume=10000000,
            ),
        ]

        count = repo.upsert_daily(data)
        assert count == 1

        result = repo.get_daily("2330", date(2024, 1, 1), date(2024, 1, 5))
        assert len(result) == 1


class TestFactorRepository:
    """Factor Repository 測試"""

    def test_create_and_get(self, session):
        repo = FactorRepository(session)

        factor = repo.create(
            name="MA5",
            expression="Mean($close, 5)",
            description="5日均線",
        )
        assert factor.id is not None

        result = repo.get_by_name("MA5")
        assert result is not None
        assert result.expression == "Mean($close, 5)"

    def test_get_active_excludes_excluded(self, session):
        repo = FactorRepository(session)

        repo.create(name="MA5", expression="Mean($close, 5)")
        factor2 = repo.create(name="MA10", expression="Mean($close, 10)")
        repo.set_excluded(factor2.id, True)

        active = repo.get_active()
        assert len(active) == 1
        assert active[0].name == "MA5"


class TestTrainingRepository:
    """Training Repository 測試"""

    def test_training_run_lifecycle(self, session):
        training_repo = TrainingRepository(session)
        factor_repo = FactorRepository(session)

        # 建立因子
        factor = factor_repo.create(name="MA5", expression="Mean($close, 5)")

        # 建立訓練
        run = training_repo.create_run(config={"model": "LGBModel"})
        assert run.id is not None
        assert run.completed_at is None

        # 新增因子結果
        training_repo.add_factor_result(
            run_id=run.id,
            factor_id=factor.id,
            ic_value=0.05,
            selected=True,
        )

        # 完成訓練
        training_repo.complete_run(run.id, model_ic=0.08)

        # 驗證
        latest = training_repo.get_latest_run()
        assert latest is not None
        assert latest.completed_at is not None
        assert float(latest.model_ic) == pytest.approx(0.08)

        selected = training_repo.get_selected_factors(run.id)
        assert len(selected) == 1
        assert float(selected[0].ic_value) == pytest.approx(0.05)
