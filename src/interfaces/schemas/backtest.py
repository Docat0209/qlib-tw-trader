"""
回測相關 Schema
"""

from pydantic import BaseModel


class BacktestRequest(BaseModel):
    """回測請求"""

    model_id: int
    initial_capital: float = 1000000.0
    max_positions: int = 10
    trade_price: str = "close"  # "close" | "open"


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
    """交易點（含盈虧資訊）"""

    date: str
    side: str  # buy / sell
    price: float
    shares: int
    amount: float | None = None
    commission: float | None = None
    pnl: float | None = None  # 賣出時的盈虧金額
    pnl_pct: float | None = None  # 賣出時的盈虧 %
    holding_days: int | None = None  # 持有天數
    stock_id: str | None = None  # 股票代碼（全局交易列表用）
    stock_name: str | None = None  # 股票名稱


class AllTradesResponse(BaseModel):
    """所有交易記錄"""

    backtest_id: int
    items: list[TradePoint]
    total_pnl: float  # 已實現盈虧
    unrealized_pnl: float = 0.0  # 未實現盈虧（持倉）
    total_equity_pnl: float = 0.0  # 總計（已實現 + 未實現）
    total: int


class StockKlineResponse(BaseModel):
    """個股 K 線回應"""

    stock_id: str
    name: str
    klines: list[KlinePoint]
    trades: list[TradePoint]
