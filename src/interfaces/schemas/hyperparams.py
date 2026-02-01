"""超參數 Schemas"""

from datetime import date, datetime

from pydantic import BaseModel


class CultivationPeriod(BaseModel):
    """培養窗口結果"""

    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    best_ic: float
    params: dict


class HyperparamsSummary(BaseModel):
    """超參數組摘要"""

    id: int
    name: str
    cultivated_at: datetime
    n_periods: int
    # 關鍵參數預覽
    learning_rate: float | None = None
    num_leaves: int | None = None


class HyperparamsDetail(BaseModel):
    """超參數組詳情"""

    id: int
    name: str
    cultivated_at: datetime
    n_periods: int
    params: dict
    stability: dict[str, float]
    periods: list[CultivationPeriod]


class HyperparamsListResponse(BaseModel):
    """超參數列表回應"""

    items: list[HyperparamsSummary]
    total: int


class CultivateRequest(BaseModel):
    """超參數培養請求"""

    name: str
    n_periods: int = 5
    n_trials_per_period: int = 20


class CultivateResponse(BaseModel):
    """超參數培養回應"""

    job_id: str
    status: str
    message: str


class HyperparamsUpdate(BaseModel):
    """更新超參數組"""

    name: str
