from datetime import datetime

from pydantic import BaseModel


class FactorBase(BaseModel):
    """因子基礎 Schema"""

    name: str
    display_name: str | None = None
    category: str = "technical"
    description: str | None = None
    formula: str  # API 中使用 formula，對應 DB 中的 expression


class FactorCreate(FactorBase):
    """新增因子"""

    pass


class FactorUpdate(BaseModel):
    """更新因子"""

    name: str | None = None
    display_name: str | None = None
    category: str | None = None
    description: str | None = None
    formula: str | None = None


class SelectionHistory(BaseModel):
    """入選歷史"""

    model_id: str
    trained_at: str
    selected: bool


class FactorResponse(BaseModel):
    """因子回應"""

    id: str
    name: str
    display_name: str | None
    category: str
    description: str | None
    formula: str
    selection_rate: float
    times_selected: int
    times_evaluated: int
    enabled: bool
    created_at: datetime


class FactorDetailResponse(FactorResponse):
    """因子詳情回應（含入選歷史）"""

    selection_history: list[SelectionHistory]


class FactorListResponse(BaseModel):
    """因子列表回應"""

    items: list[FactorResponse]
    total: int


class ValidateRequest(BaseModel):
    """驗證請求"""

    expression: str


class ValidateResponse(BaseModel):
    """驗證回應"""

    valid: bool
    error: str | None = None
    fields_used: list[str] = []
    operators_used: list[str] = []
    warnings: list[str] = []


class SeedResponse(BaseModel):
    """Seed 回應"""

    success: bool
    inserted: int
    message: str


class AvailableFieldsResponse(BaseModel):
    """可用欄位回應"""

    fields: list[str]
    operators: list[str]


class DeduplicateResponse(BaseModel):
    """因子去重回應"""

    success: bool
    total_factors: int
    kept_factors: int
    disabled_factors: int
    disabled_names: list[str]
    message: str
