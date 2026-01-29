from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.database import Base

TZ_TAIPEI = ZoneInfo("Asia/Taipei")


def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


class StockDaily(Base):
    """日K線資料"""

    __tablename__ = "stock_daily"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_date"),
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


class Factor(Base):
    """因子定義"""

    __tablename__ = "factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    expression: Mapped[str] = mapped_column(Text)  # qlib ExpressionOps 語法
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)  # 永久排除
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)


class TrainingRun(Base):
    """訓練執行記錄"""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now_taipei)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    model_ic: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 格式

    # 關聯
    selected_factors: Mapped[list["TrainingFactorResult"]] = relationship(
        back_populates="training_run"
    )


class TrainingFactorResult(Base):
    """訓練結果中的因子表現"""

    __tablename__ = "training_factor_results"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "factor_id", name="uq_training_factor"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_runs.id")
    )
    factor_id: Mapped[int] = mapped_column(Integer, ForeignKey("factors.id"))
    ic_value: Mapped[float] = mapped_column(Numeric(10, 6))
    selected: Mapped[bool] = mapped_column(Boolean)  # 是否被選入最終模型

    # 關聯
    training_run: Mapped["TrainingRun"] = relationship(
        back_populates="selected_factors"
    )
    factor: Mapped["Factor"] = relationship()
