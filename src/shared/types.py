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
    total_shares: int               # 發行股數
    foreign_shares: int             # 外資持股數
    foreign_ratio: Decimal          # 外資持股比率
    foreign_remaining_shares: int   # 尚可投資股數
    foreign_remaining_ratio: Decimal  # 尚可投資比率
    foreign_upper_limit_ratio: Decimal  # 外資投資上限比率
    chinese_upper_limit_ratio: Decimal  # 陸資投資上限比率


# === 低頻 ===

@dataclass
class MonthlyRevenue:
    """月營收"""
    stock_id: str
    year: int
    month: int
    revenue: Decimal


# === 事件型 ===

@dataclass
class SecuritiesLending:
    """借券明細（每日聚合）"""
    date: date
    stock_id: str
    lending_volume: int       # 當日借券成交量（張）
