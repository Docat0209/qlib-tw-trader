from dataclasses import dataclass
from datetime import date
from decimal import Decimal


# === 個股日頻 ===

@dataclass
class OHLCV:
    """日K線資料"""
    date: date
    stock_id: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass
class AdjClose:
    """還原收盤價"""
    date: date
    stock_id: str
    adj_close: Decimal


@dataclass
class PER:
    """PER/PBR/殖利率"""
    date: date
    stock_id: str
    pe_ratio: Decimal | None
    pb_ratio: Decimal | None
    dividend_yield: Decimal | None


@dataclass
class Institutional:
    """三大法人買賣超"""
    date: date
    stock_id: str
    foreign_buy: int
    foreign_sell: int
    trust_buy: int
    trust_sell: int
    dealer_buy: int
    dealer_sell: int

    @property
    def foreign_net(self) -> int:
        return self.foreign_buy - self.foreign_sell

    @property
    def trust_net(self) -> int:
        return self.trust_buy - self.trust_sell

    @property
    def dealer_net(self) -> int:
        return self.dealer_buy - self.dealer_sell


@dataclass
class Margin:
    """融資融券"""
    date: date
    stock_id: str
    margin_buy: int
    margin_sell: int
    margin_balance: int
    short_buy: int
    short_sell: int
    short_balance: int


@dataclass
class Shareholding:
    """外資持股"""
    date: date
    stock_id: str
    foreign_shares: int
    foreign_ratio: Decimal


# === 市場日頻 ===

@dataclass
class MarketInstitutional:
    """整體三大法人"""
    date: date
    foreign_buy: Decimal
    foreign_sell: Decimal
    trust_buy: Decimal
    trust_sell: Decimal
    dealer_buy: Decimal
    dealer_sell: Decimal


@dataclass
class MarketMargin:
    """整體融資融券"""
    date: date
    margin_balance: Decimal
    short_balance: Decimal


# === 低頻 ===

@dataclass
class MonthlyRevenue:
    """月營收"""
    stock_id: str
    year: int
    month: int
    revenue: Decimal
    revenue_yoy: Decimal | None
    revenue_mom: Decimal | None


@dataclass
class QuarterlyFinancial:
    """季度財報（綜合損益表）"""
    stock_id: str
    year: int
    quarter: int
    revenue: Decimal | None
    gross_profit: Decimal | None
    operating_income: Decimal | None
    net_income: Decimal | None
    eps: Decimal | None


@dataclass
class QuarterlyBalance:
    """資產負債表"""
    stock_id: str
    year: int
    quarter: int
    total_assets: Decimal | None
    total_liabilities: Decimal | None
    total_equity: Decimal | None


@dataclass
class QuarterlyCashFlow:
    """現金流量表"""
    stock_id: str
    year: int
    quarter: int
    operating_cf: Decimal | None
    investing_cf: Decimal | None
    financing_cf: Decimal | None
    free_cf: Decimal | None


# === 事件型 ===

@dataclass
class Dividend:
    """除權息"""
    stock_id: str
    ex_date: date
    cash_dividend: Decimal | None
    stock_dividend: Decimal | None
