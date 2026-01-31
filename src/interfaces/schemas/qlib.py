"""Qlib 導出 API 的 Pydantic 模型"""

from datetime import date

from pydantic import BaseModel, Field


class ExportRequest(BaseModel):
    """導出請求"""

    start_date: date = Field(description="開始日期")
    end_date: date = Field(description="結束日期")
    include_fields: list[str] | None = Field(
        default=None,
        description="要導出的欄位（不指定則全部導出）",
    )


class ExportResponse(BaseModel):
    """導出回應"""

    job_id: str = Field(description="任務 ID")
    message: str = Field(description="訊息")


class ExportResultResponse(BaseModel):
    """導出結果"""

    stocks_exported: int = Field(description="導出股票數")
    fields_per_stock: int = Field(description="每股票欄位數")
    total_files: int = Field(description="總檔案數")
    calendar_days: int = Field(description="交易日數")
    output_path: str = Field(description="輸出路徑")
    errors: list[dict] = Field(description="錯誤清單")


class ValidateResponse(BaseModel):
    """驗證回應"""

    stock_id: str = Field(description="股票代碼")
    field: str = Field(description="欄位名稱")
    file_exists: bool = Field(description="檔案是否存在")
    record_count: int = Field(description="記錄數")
    nan_count: int = Field(description="NaN 數量")
    sample_data: list[dict] | None = Field(default=None, description="取樣資料")


class FieldInfo(BaseModel):
    """欄位資訊"""

    name: str = Field(description="欄位名稱")
    source: str = Field(description="資料來源")
    attribute: str = Field(description="屬性名稱")


class FieldsResponse(BaseModel):
    """欄位清單回應"""

    fields: list[FieldInfo] = Field(description="可導出欄位清單")
    total: int = Field(description="總欄位數")


class StatusResponse(BaseModel):
    """狀態回應"""

    exists: bool = Field(description="導出目錄是否存在")
    stocks: int = Field(description="股票數")
    calendar_days: int = Field(description="交易日數")
    output_path: str = Field(description="輸出路徑")
