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


class ModelResponse(BaseModel):
    """模型回應"""

    id: str
    trained_at: datetime
    factor_count: int | None
    factors: list[str]
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    training_duration_seconds: int | None
    is_current: bool


class ModelHistoryItem(BaseModel):
    """歷史模型項目"""

    id: str
    trained_at: datetime
    factor_count: int | None
    train_period: Period | None
    valid_period: Period | None
    metrics: ModelMetrics
    is_current: bool


class ModelHistoryResponse(BaseModel):
    """歷史模型列表回應"""

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

    train_end: date
    valid_end: date


class TrainResponse(BaseModel):
    """訓練回應"""

    job_id: str
    status: str
    message: str
