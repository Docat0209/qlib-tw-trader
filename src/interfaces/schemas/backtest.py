"""
回測相關 Schema
"""

from pydantic import BaseModel


class BacktestRequest(BaseModel):
    """回測請求"""

    model_id: int
    initial_capital: float = 1000000.0
    max_positions: int = 10


class BacktestMetrics(BaseModel):
    """回測績效指標"""

    total_return_with_cost: float | None = None
    total_return_without_cost: float | None = None
    annual_return_with_cost: float | None = None
    annual_return_without_cost: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None
    total_cost: float | None = None

    # 向後兼容舊欄位
    total_return: float | None = None
    annual_return: float | None = None
    profit_factor: float | None = None


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


# === 新增：股票交易 API ===


class StockTradeInfo(BaseModel):
    """股票交易摘要"""

    stock_id: str
    name: str
    buy_count: int
    sell_count: int
    total_pnl: float | None = None


class StockTradeListResponse(BaseModel):
    """股票交易清單"""

    backtest_id: int
    items: list[StockTradeInfo]
    total: int


class KlinePoint(BaseModel):
    """K 線資料點"""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TradePoint(BaseModel):
    """交易點"""

    date: str
    side: str  # buy / sell
    price: float
    shares: int


class StockKlineResponse(BaseModel):
    """個股 K 線回應"""

    stock_id: str
    name: str
    klines: list[KlinePoint]
    trades: list[TradePoint]
