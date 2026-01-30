"""
日頻資料 Repository 實作
"""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from src.repositories.base import MarketDailyRepository, StockDailyRepository
from src.repositories.models import (
    MarketDailyInstitutional,
    MarketDailyMargin,
    StockDaily,
    StockDailyAdj,
    StockDailyInstitutional,
    StockDailyMargin,
    StockDailyPER,
    StockDailyShareholding,
)
from src.shared.types import (
    AdjClose,
    Institutional,
    Margin,
    MarketInstitutional,
    MarketMargin,
    OHLCV,
    PER,
    Shareholding,
)


# =============================================================================
# 個股日頻
# =============================================================================


class OHLCVRepository(StockDailyRepository[OHLCV, StockDaily]):
    """日K線 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDaily)

    def _to_dataclass(self, row: StockDaily) -> OHLCV:
        return OHLCV(
            date=row.date,
            stock_id=row.stock_id,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )

    def _to_dict(self, data: OHLCV) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]


class AdjCloseRepository(StockDailyRepository[AdjClose, StockDailyAdj]):
    """還原股價 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDailyAdj)

    def _to_dataclass(self, row: StockDailyAdj) -> AdjClose:
        return AdjClose(
            date=row.date,
            stock_id=row.stock_id,
            adj_close=row.adj_close,
        )

    def _to_dict(self, data: AdjClose) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "adj_close": data.adj_close,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["adj_close"]


class PERRepository(StockDailyRepository[PER, StockDailyPER]):
    """PER/PBR Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDailyPER)

    def _to_dataclass(self, row: StockDailyPER) -> PER:
        return PER(
            date=row.date,
            stock_id=row.stock_id,
            pe_ratio=row.pe_ratio,
            pb_ratio=row.pb_ratio,
            dividend_yield=row.dividend_yield,
        )

    def _to_dict(self, data: PER) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "pe_ratio": data.pe_ratio,
            "pb_ratio": data.pb_ratio,
            "dividend_yield": data.dividend_yield,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["pe_ratio", "pb_ratio", "dividend_yield"]


class InstitutionalRepository(StockDailyRepository[Institutional, StockDailyInstitutional]):
    """三大法人 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDailyInstitutional)

    def _to_dataclass(self, row: StockDailyInstitutional) -> Institutional:
        return Institutional(
            date=row.date,
            stock_id=row.stock_id,
            foreign_buy=row.foreign_buy,
            foreign_sell=row.foreign_sell,
            trust_buy=row.trust_buy,
            trust_sell=row.trust_sell,
            dealer_buy=row.dealer_buy,
            dealer_sell=row.dealer_sell,
        )

    def _to_dict(self, data: Institutional) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "foreign_buy": data.foreign_buy,
            "foreign_sell": data.foreign_sell,
            "trust_buy": data.trust_buy,
            "trust_sell": data.trust_sell,
            "dealer_buy": data.dealer_buy,
            "dealer_sell": data.dealer_sell,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["foreign_buy", "foreign_sell", "trust_buy", "trust_sell", "dealer_buy", "dealer_sell"]


class MarginRepository(StockDailyRepository[Margin, StockDailyMargin]):
    """融資融券 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDailyMargin)

    def _to_dataclass(self, row: StockDailyMargin) -> Margin:
        return Margin(
            date=row.date,
            stock_id=row.stock_id,
            margin_buy=row.margin_buy,
            margin_sell=row.margin_sell,
            margin_balance=row.margin_balance,
            short_buy=row.short_buy,
            short_sell=row.short_sell,
            short_balance=row.short_balance,
        )

    def _to_dict(self, data: Margin) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "margin_buy": data.margin_buy,
            "margin_sell": data.margin_sell,
            "margin_balance": data.margin_balance,
            "short_buy": data.short_buy,
            "short_sell": data.short_sell,
            "short_balance": data.short_balance,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["margin_buy", "margin_sell", "margin_balance", "short_buy", "short_sell", "short_balance"]


class ShareholdingRepository(StockDailyRepository[Shareholding, StockDailyShareholding]):
    """外資持股 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, StockDailyShareholding)

    def _to_dataclass(self, row: StockDailyShareholding) -> Shareholding:
        return Shareholding(
            date=row.date,
            stock_id=row.stock_id,
            foreign_shares=row.foreign_shares,
            foreign_ratio=row.foreign_ratio,
        )

    def _to_dict(self, data: Shareholding) -> dict:
        return {
            "stock_id": data.stock_id,
            "date": data.date,
            "foreign_shares": data.foreign_shares,
            "foreign_ratio": data.foreign_ratio,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["stock_id", "date"]

    def _get_update_fields(self) -> list[str]:
        return ["foreign_shares", "foreign_ratio"]


# =============================================================================
# 市場日頻
# =============================================================================


class MarketInstitutionalRepository(MarketDailyRepository[MarketInstitutional, MarketDailyInstitutional]):
    """整體三大法人 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, MarketDailyInstitutional)

    def _to_dataclass(self, row: MarketDailyInstitutional) -> MarketInstitutional:
        return MarketInstitutional(
            date=row.date,
            foreign_buy=row.foreign_buy,
            foreign_sell=row.foreign_sell,
            trust_buy=row.trust_buy,
            trust_sell=row.trust_sell,
            dealer_buy=row.dealer_buy,
            dealer_sell=row.dealer_sell,
        )

    def _to_dict(self, data: MarketInstitutional) -> dict:
        return {
            "date": data.date,
            "foreign_buy": data.foreign_buy,
            "foreign_sell": data.foreign_sell,
            "trust_buy": data.trust_buy,
            "trust_sell": data.trust_sell,
            "dealer_buy": data.dealer_buy,
            "dealer_sell": data.dealer_sell,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["date"]

    def _get_update_fields(self) -> list[str]:
        return ["foreign_buy", "foreign_sell", "trust_buy", "trust_sell", "dealer_buy", "dealer_sell"]


class MarketMarginRepository(MarketDailyRepository[MarketMargin, MarketDailyMargin]):
    """整體融資融券 Repository"""

    def __init__(self, session: Session):
        super().__init__(session, MarketDailyMargin)

    def _to_dataclass(self, row: MarketDailyMargin) -> MarketMargin:
        return MarketMargin(
            date=row.date,
            margin_balance=row.margin_balance,
            short_balance=row.short_balance,
        )

    def _to_dict(self, data: MarketMargin) -> dict:
        return {
            "date": data.date,
            "margin_balance": data.margin_balance,
            "short_balance": data.short_balance,
        }

    def _get_conflict_keys(self) -> list[str]:
        return ["date"]

    def _get_update_fields(self) -> list[str]:
        return ["margin_balance", "short_balance"]


