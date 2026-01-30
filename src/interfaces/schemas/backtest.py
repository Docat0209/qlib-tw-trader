"""
回測相關 Schema
"""

from datetime import date, datetime

from pydantic import BaseModel


class BacktestRequest(BaseModel):
    """回測請求"""

    model_id: int
    start_date: date
    end_date: date
    initial_capital: float = 1000000.0
    max_positions: int = 10


class BacktestMetrics(BaseModel):
    """回測績效指標"""

    total_return: float | None
    annual_return: float | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    total_trades: int | None


class EquityCurvePoint(BaseModel):
    """權益曲線點"""

    date: str
    equity: float
    benchmark: float | None = None
    drawdown: float | None = None


class BacktestResponse(BaseModel):
    """回測回應"""

    id: int
    model_id: int
    start_date: str
    end_date: str
    initial_capital: float
    max_positions: int
    status: str
    metrics: BacktestMetrics | None = None
    created_at: str


class BacktestDetailResponse(BacktestResponse):
    """回測詳情"""

    equity_curve: list[EquityCurvePoint] | None = None


class BacktestListResponse(BaseModel):
    """回測列表"""

    items: list[BacktestResponse]
    total: int


class BacktestRunResponse(BaseModel):
    """觸發回測回應"""

    backtest_id: int
    job_id: str
    status: str
    message: str
