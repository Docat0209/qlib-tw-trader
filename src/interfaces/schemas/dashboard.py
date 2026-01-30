from pydantic import BaseModel


class FactorsSummary(BaseModel):
    """因子摘要"""

    total: int
    enabled: int
    low_selection_count: int


class ModelSummary(BaseModel):
    """模型摘要"""

    last_trained_at: str | None
    days_since_training: int | None
    needs_retrain: bool
    factor_count: int | None
    ic: float | None
    icir: float | None


class TopPick(BaseModel):
    """首選股票"""

    symbol: str
    score: float


class PredictionSummary(BaseModel):
    """預測摘要"""

    date: str | None
    buy_signals: int
    sell_signals: int
    top_pick: TopPick | None


class DataStatusSummary(BaseModel):
    """資料狀態摘要"""

    is_complete: bool
    last_updated: str | None
    missing_count: int


class PerformanceSummaryBrief(BaseModel):
    """績效摘要（簡易）"""

    today_return: float | None
    mtd_return: float | None
    ytd_return: float | None
    total_return: float | None


class DashboardSummary(BaseModel):
    """Dashboard 摘要"""

    factors: FactorsSummary
    model: ModelSummary
    prediction: PredictionSummary
    data_status: DataStatusSummary
    performance: PerformanceSummaryBrief
