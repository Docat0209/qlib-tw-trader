from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass
class OHLCV:
    """日K線資料"""
    date: date
    stock_id: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int  # 成交股數
