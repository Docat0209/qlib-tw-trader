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

    # 核心指標
    total_return_with_cost: float | None = None
    total_return_without_cost: float | None = None
    annual_return_with_cost: float | None = None
    annual_return_without_cost: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None
    total_cost: float | None = None

    # 市場基準
    market_return: float | None = None  # 所有股票的平均報酬
    market_hit_rate: float | None = None  # 上漲股票比例（隨機選股的預期勝率）
    market_stocks_up: int | None = None  # 上漲股票數
    market_stocks_down: int | None = None  # 下跌股票數

    # 超額表現
    excess_return: float | None = None  # 模型報酬 - 市場報酬
    excess_hit_rate: float | None = None  # 模型勝率 - 市場勝率
    alpha: float | None = None  # 與 excess_return 相同（標準化命名）

    # 風險調整指標
    sortino_ratio: float | None = None  # excess_return / downside_std
    information_ratio: float | None = None  # alpha / tracking_error
    calmar_ratio: float | None = None  # annual_return / max_drawdown

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


# === 多期統計 ===


class PeriodSummary(BaseModel):
    """單期摘要"""

    period: str  # YYYYMM
    model_return: float
    market_return: float
    excess_return: float
    win_rate: float
    market_hit_rate: float
    beat_market: bool  # 是否跑贏市場


class BacktestSummary(BaseModel):
    """多期累積統計"""

    selection_method: str | None = None  # 因子選擇方法（如 composite_threshold_v3）
    n_periods: int  # 統計期數
    cumulative_return: float  # 累積報酬
    cumulative_excess_return: float  # 累積超額報酬
    avg_period_return: float  # 平均單期報酬
    avg_excess_return: float  # 平均超額報酬
    period_win_rate: float  # 跑贏市場期數比例
    return_std: float  # 報酬標準差
    excess_return_std: float  # 超額報酬標準差
    t_statistic: float | None = None  # t 統計量
    p_value: float | None = None  # p 值
    ci_lower: float | None = None  # 95% CI 下限
    ci_upper: float | None = None  # 95% CI 上限
    is_significant: bool  # p_value < 0.05
    periods: list[PeriodSummary]  # 各期詳情


# =============================================================================
# Walk-Forward 回測
# =============================================================================


class WalkForwardRequest(BaseModel):
    """Walk-Forward 回測請求"""

    start_week_id: str  # "2024W01"
    end_week_id: str  # "2025W20"
    initial_capital: float = 1000000.0
    max_positions: int = 10
    trade_price: str = "open"
    enable_incremental: bool = False
    strategy: str = "topk"
    hyperparams_id: int | None = None  # 指定超參數組 ID


class WalkForwardRunResponse(BaseModel):
    """觸發 Walk-Forward 回測回應"""

    backtest_id: int
    job_id: str
    status: str
    message: str


class WeekStatus(BaseModel):
    """週狀態"""

    week_id: str
    status: str  # "available" | "missing" | "not_allowed"
    model_name: str | None = None
    valid_ic: float | None = None
    fallback_week: str | None = None
    fallback_model: str | None = None
    reason: str | None = None  # 不可選的原因


class AvailableWeeksResponse(BaseModel):
    """可回測週列表"""

    weeks: list[WeekStatus]
    current_week_id: str


class IcAnalysis(BaseModel):
    """IC 分析結果"""

    avg_valid_ic: float
    avg_live_ic: float
    ic_decay: float  # (valid - live) / valid * 100
    ic_correlation: float | None = None  # 驗證 IC 與實盤 IC 的相關係數


class WalkForwardReturnMetrics(BaseModel):
    """Walk-Forward 收益指標"""

    cumulative_return: float
    market_return: float
    excess_return: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None


class WeeklyDetail(BaseModel):
    """週詳情"""

    predict_week: str  # 預測的週 (e.g., "2024W02")
    model_week: str  # 使用的模型週 (e.g., "2024W01")
    model_name: str  # 模型名稱
    valid_ic: float | None = None  # 驗證期 IC
    live_ic: float | None = None  # 實盤 IC
    ic_decay: float | None = None  # IC 衰減 %
    week_return: float | None = None  # 週收益 %
    market_return: float | None = None  # 市場收益 %
    is_fallback: bool = False  # 是否使用 fallback 模型
    incremental_days: int | None = None  # 增量學習使用的天數


class WalkForwardConfig(BaseModel):
    """Walk-Forward 回測配置"""

    initial_capital: float
    max_positions: int
    trade_price: str
    enable_incremental: bool
    strategy: str


class WalkForwardResponse(BaseModel):
    """Walk-Forward 回測回應"""

    id: int
    start_week_id: str
    end_week_id: str
    status: str
    config: WalkForwardConfig
    created_at: str
    completed_at: str | None = None


class WalkForwardDetailResponse(WalkForwardResponse):
    """Walk-Forward 回測詳情"""

    ic_analysis: IcAnalysis | None = None
    return_metrics: WalkForwardReturnMetrics | None = None
    weekly_details: list[WeeklyDetail] | None = None
    equity_curve: list[EquityCurvePoint] | None = None


class WalkForwardListResponse(BaseModel):
    """Walk-Forward 回測列表"""

    items: list[WalkForwardResponse]
    total: int
