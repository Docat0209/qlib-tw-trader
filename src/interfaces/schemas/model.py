from datetime import date, datetime

from pydantic import BaseModel


class Period(BaseModel):
    """期間"""

    start: date
    end: date


class ModelMetrics(BaseModel):
    """模型指標"""

    ic: float | None
    icir: float | None


class FactorSummary(BaseModel):
    """因子摘要（用於模型詳情）"""

    id: str
    name: str
    display_name: str | None
    category: str
    ic_value: float | None = None  # 該因子在此模型中的 IC 值


class ModelResponse(BaseModel):
    """模型詳情回應"""

    id: str
    name: str | None  # 訓練區間格式：2022-01~2025-01
    description: str | None = None
    status: str  # queued / running / completed / failed
    trained_at: datetime
    factor_count: int | None
    factors: list[str]  # 向後兼容：選中因子名稱列表
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    training_duration_seconds: int | None
    is_current: bool
    candidate_factors: list[FactorSummary] = []  # 候選因子
    selected_factors: list[FactorSummary] = []  # 選中因子


class ModelSummary(BaseModel):
    """模型列表項目"""

    id: str
    name: str | None
    status: str
    trained_at: datetime
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    factor_count: int | None  # 選中因子數
    candidate_count: int | None  # 候選因子數
    is_current: bool


class ModelListResponse(BaseModel):
    """模型列表回應"""

    items: list[ModelSummary]
    total: int


# 向後兼容的別名
ModelHistoryItem = ModelSummary


class ModelHistoryResponse(BaseModel):
    """歷史模型列表回應（向後兼容）"""

    items: list[ModelHistoryItem]
    total: int


class ModelComparisonItem(BaseModel):
    """模型比較項目"""

    id: str
    trained_at: str
    ic: float | None
    icir: float | None
    factor_count: int | None


class ModelComparisonResponse(BaseModel):
    """模型比較回應"""

    models: list[ModelComparisonItem]


class ModelStatus(BaseModel):
    """訓練狀態"""

    last_trained_at: datetime | None
    days_since_training: int | None
    needs_retrain: bool
    retrain_threshold_days: int
    current_job: int | None


class TrainRequest(BaseModel):
    """訓練請求"""

    train_end: date | None = None  # 預設：今日 - 驗證期長度


class TrainResponse(BaseModel):
    """訓練回應"""

    job_id: str
    status: str
    message: str
