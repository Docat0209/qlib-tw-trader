"""資料來源 Adapters"""

from src.adapters.base import (
    BaseAdapter,
    BulkDataAdapter,
    MarketDataAdapter,
    StockDataAdapter,
)
from src.adapters.finmind import (
    FinMindBalanceAdapter,
    FinMindCashFlowAdapter,
    FinMindDividendAdapter,
    FinMindFinancialAdapter,
    FinMindInstitutionalAdapter,
    FinMindMarginAdapter,
    FinMindMonthlyRevenueAdapter,
    FinMindOHLCVAdapter,
    FinMindPERAdapter,
    FinMindSecuritiesLendingAdapter,
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

__all__ = [
    # Base
    "BaseAdapter",
    "BulkDataAdapter",
    "MarketDataAdapter",
    "StockDataAdapter",
    # TWSE
    "TwseBulkOHLCVAdapter",
    "TwseBulkPERAdapter",
    "TwseBulkInstitutionalAdapter",
    "TwseBulkMarginAdapter",
    "TwseBulkShareholdingAdapter",
    "TwseStockOHLCVAdapter",
    # FinMind
    "FinMindOHLCVAdapter",
    "FinMindPERAdapter",
    "FinMindInstitutionalAdapter",
    "FinMindMarginAdapter",
    "FinMindShareholdingAdapter",
    "FinMindSecuritiesLendingAdapter",
    "FinMindMonthlyRevenueAdapter",
    "FinMindFinancialAdapter",
    "FinMindBalanceAdapter",
    "FinMindCashFlowAdapter",
    "FinMindDividendAdapter",
    # yfinance
    "YFinanceAdjCloseAdapter",
]
