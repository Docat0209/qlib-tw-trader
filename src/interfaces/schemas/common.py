"""
共用 Schema
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Meta(BaseModel):
    """分頁元資料"""

    total: int
    page: int
    page_size: int
    pages: int


class PaginatedResponse(BaseModel, Generic[T]):
    """分頁回應"""

    data: list[T]
    meta: Meta


class ErrorDetail(BaseModel):
    """錯誤詳情"""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """錯誤回應"""

    error: ErrorDetail
