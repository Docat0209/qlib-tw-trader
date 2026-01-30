"""資料存取 Repositories"""

from src.repositories.database import Base, engine, get_session, init_db
from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    MarketInstitutionalRepository,
    MarketMarginRepository,
    OHLCVRepository,
    PERRepository,
    SecuritiesLendingRepository,
    ShareholdingRepository,
)
from src.repositories.periodic import (
    DividendRepository,
    MonthlyRevenueRepository,
    QuarterlyBalanceRepository,
    QuarterlyCashFlowRepository,
    QuarterlyFinancialRepository,
)

__all__ = [
    # Database
    "Base",
    "engine",
    "get_session",
    "init_db",
    # Stock Daily
    "OHLCVRepository",
    "AdjCloseRepository",
    "PERRepository",
    "InstitutionalRepository",
    "MarginRepository",
    "ShareholdingRepository",
    "SecuritiesLendingRepository",
    # Market Daily
    "MarketInstitutionalRepository",
    "MarketMarginRepository",
    # Periodic
    "MonthlyRevenueRepository",
    "QuarterlyFinancialRepository",
    "QuarterlyBalanceRepository",
    "QuarterlyCashFlowRepository",
    "DividendRepository",
]
