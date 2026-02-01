from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.database import Base

# Note: StockDailyInstitutional 已存在（三大法人）

TZ_TAIPEI = ZoneInfo("Asia/Taipei")


def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


# =============================================================================
# 個股日頻
# =============================================================================

class StockDaily(Base):
    """日K線資料"""

    __tablename__ = "stock_daily"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    high: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    low: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    close: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    volume: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailyAdj(Base):
    """還原股價"""

    __tablename__ = "stock_daily_adj"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_adj"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    adj_close: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailyPER(Base):
    """PER/PBR/殖利率"""

    __tablename__ = "stock_daily_per"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_per"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    pe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    pb_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    dividend_yield: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailyInstitutional(Base):
    """三大法人買賣超"""

    __tablename__ = "stock_daily_institutional"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_inst"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    foreign_buy: Mapped[int] = mapped_column(Integer)
    foreign_sell: Mapped[int] = mapped_column(Integer)
    trust_buy: Mapped[int] = mapped_column(Integer)
    trust_sell: Mapped[int] = mapped_column(Integer)
    dealer_buy: Mapped[int] = mapped_column(Integer)
    dealer_sell: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailyMargin(Base):
    """融資融券"""

    __tablename__ = "stock_daily_margin"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_margin"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    margin_buy: Mapped[int] = mapped_column(Integer)
    margin_sell: Mapped[int] = mapped_column(Integer)
    margin_balance: Mapped[int] = mapped_column(Integer)
    short_buy: Mapped[int] = mapped_column(Integer)
    short_sell: Mapped[int] = mapped_column(Integer)
    short_balance: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailyShareholding(Base):
    """外資持股"""

    __tablename__ = "stock_daily_shareholding"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_share"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    # 發行股數
    total_shares: Mapped[int] = mapped_column(Integer)
    # 外資持股
    foreign_shares: Mapped[int] = mapped_column(Integer)
    foreign_ratio: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    # 尚可投資
    foreign_remaining_shares: Mapped[int] = mapped_column(Integer)
    foreign_remaining_ratio: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    # 投資上限
    foreign_upper_limit_ratio: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    chinese_upper_limit_ratio: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockDailySecuritiesLending(Base):
    """借券明細（每日聚合）"""

    __tablename__ = "stock_daily_securities_lending"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_daily_sl"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    lending_volume: Mapped[int] = mapped_column(Integer)  # 當日借券成交量（張）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


# =============================================================================
# 低頻
# =============================================================================

class StockMonthlyRevenue(Base):
    """月營收"""

    __tablename__ = "stock_monthly_revenue"
    __table_args__ = (
        UniqueConstraint("stock_id", "year", "month", name="uq_stock_monthly_rev"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    revenue: Mapped[Decimal] = mapped_column(Numeric(16, 0))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


# =============================================================================
# 其他
# =============================================================================


class TradingCalendar(Base):
    """交易日曆"""

    __tablename__ = "trading_calendar"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_trading_day: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class StockUniverse(Base):
    """股票池"""

    __tablename__ = "stock_universe"

    stock_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    market_cap: Mapped[int] = mapped_column(Integer)  # 市值（億）
    rank: Mapped[int] = mapped_column(Integer)  # 市值排名
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


# =============================================================================
# 因子 & 訓練（已存在）
# =============================================================================

class Factor(Base):
    """因子定義"""

    __tablename__ = "factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str] = mapped_column(String(20), default="technical")
    expression: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class TrainingRun(Base):
    """訓練執行記錄"""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 2022-01~2025-01
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    train_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    train_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    model_ic: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    icir: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    factor_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    config: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 因子池記錄（JSON 格式的 ID 列表）
    candidate_factor_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    selected_factor_ids: Mapped[str | None] = mapped_column(Text, nullable=True)

    selected_factors: Mapped[list["TrainingFactorResult"]] = relationship(
        back_populates="training_run"
    )


class TrainingFactorResult(Base):
    """訓練結果中的因子表現"""

    __tablename__ = "training_factor_results"
    __table_args__ = (
        UniqueConstraint("training_run_id", "factor_id", name="uq_training_factor"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_runs.id"))
    factor_id: Mapped[int] = mapped_column(Integer, ForeignKey("factors.id"))
    ic_value: Mapped[float] = mapped_column(Numeric(10, 6))
    selected: Mapped[bool] = mapped_column(Boolean)

    training_run: Mapped["TrainingRun"] = relationship(back_populates="selected_factors")
    factor: Mapped["Factor"] = relationship()


# =============================================================================
# 交易記錄
# =============================================================================


class Trade(Base):
    """交易記錄"""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("backtests.id"), nullable=True, index=True
    )
    date: Mapped[date] = mapped_column(Date, index=True)
    stock_id: Mapped[str] = mapped_column(String(10), index=True)
    side: Mapped[str] = mapped_column(String(10))  # buy/sell
    shares: Mapped[int] = mapped_column(Integer)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2))
    commission: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


# =============================================================================
# 非同步任務
# =============================================================================


class Job(Base):
    """非同步任務"""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    job_type: Mapped[str] = mapped_column(String(20))  # train/backtest/sync
    status: Mapped[str] = mapped_column(String(20), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Backtest(Base):
    """回測記錄"""

    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_runs.id"))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(16, 2))
    max_positions: Mapped[int] = mapped_column(Integer, default=10)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    equity_curve: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    status: Mapped[str] = mapped_column(String(20), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


# =============================================================================
# 超參數
# =============================================================================


class Hyperparams(Base):
    """超參數組"""

    __tablename__ = "hyperparams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    cultivated_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)
    n_periods: Mapped[int] = mapped_column(Integer)
    params_json: Mapped[str] = mapped_column(Text)  # LightGBM 超參數 JSON
    stability_json: Mapped[str] = mapped_column(Text)  # 穩定性指標 JSON
    periods_json: Mapped[str] = mapped_column(Text)  # 各窗口結果 JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)
