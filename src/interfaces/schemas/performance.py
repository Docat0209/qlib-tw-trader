from pydantic import BaseModel, Field


class Returns(BaseModel):
    """收益數據"""

    today: float | None
    wtd: float | None
    mtd: float | None
    ytd: float | None
    total: float | None


class Alpha(BaseModel):
    """超額收益"""

    mtd: float | None
    ytd: float | None


class PerformanceSummary(BaseModel):
    """績效摘要"""

    as_of: str
    returns: Returns
    benchmark_returns: Returns
    alpha: Alpha


class EquityCurvePoint(BaseModel):
    """權益曲線點"""

    date: str
    portfolio_value: float
    cumulative_return: float
    benchmark_return: float | None


class EquityCurveResponse(BaseModel):
    """權益曲線回應"""

    data: list[EquityCurvePoint]


class MonthlyReturn(BaseModel):
    """月報酬"""

    year: int
    month: int
    return_: float = Field(serialization_alias="return")  # 'return' is reserved
    benchmark: float | None


class MonthlyReturnsResponse(BaseModel):
    """月報酬回應"""

    data: list[MonthlyReturn]
