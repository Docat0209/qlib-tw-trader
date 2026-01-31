"""Qlib 導出 API 路由"""

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.qlib import (
    ExportRequest,
    ExportResponse,
    ExportResultResponse,
    FieldInfo,
    FieldsResponse,
    StatusResponse,
    ValidateResponse,
)
from src.services.job_manager import job_manager
from src.services.qlib_exporter import ExportConfig, QlibExporter

router = APIRouter(prefix="/qlib", tags=["qlib"])

QLIB_OUTPUT_DIR = Path("data/qlib")


@router.post("/export", response_model=ExportResponse)
async def export_qlib(
    request: ExportRequest,
    session: Session = Depends(get_db),
):
    """
    導出 Qlib .bin 格式資料

    這是一個非同步任務，會返回 job_id 供追蹤進度。
    """

    async def export_task(progress_callback, **kwargs):
        exporter = QlibExporter(session)
        config = ExportConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            output_dir=QLIB_OUTPUT_DIR,
            include_fields=request.include_fields,
        )

        await progress_callback(10, "Loading trading calendar...")
        result = exporter.export(config)
        await progress_callback(100, "Export completed")

        return {
            "stocks_exported": result.stocks_exported,
            "fields_per_stock": result.fields_per_stock,
            "total_files": result.total_files,
            "calendar_days": result.calendar_days,
            "output_path": result.output_path,
            "errors": result.errors,
        }

    job_id = await job_manager.create_job(
        job_type="qlib_export",
        task_fn=export_task,
        message=f"Exporting {request.start_date} ~ {request.end_date}",
    )

    return ExportResponse(
        job_id=job_id,
        message=f"Export job created: {request.start_date} ~ {request.end_date}",
    )


@router.post("/export/sync", response_model=ExportResultResponse)
def export_qlib_sync(
    request: ExportRequest,
    session: Session = Depends(get_db),
):
    """
    同步導出 Qlib .bin 格式資料

    直接執行導出，適合小規模資料或測試用途。
    """
    exporter = QlibExporter(session)
    config = ExportConfig(
        start_date=request.start_date,
        end_date=request.end_date,
        output_dir=QLIB_OUTPUT_DIR,
        include_fields=request.include_fields,
    )

    result = exporter.export(config)

    return ExportResultResponse(
        stocks_exported=result.stocks_exported,
        fields_per_stock=result.fields_per_stock,
        total_files=result.total_files,
        calendar_days=result.calendar_days,
        output_path=result.output_path,
        errors=result.errors,
    )


@router.get("/validate/{stock_id}/{field}", response_model=ValidateResponse)
def validate_bin(stock_id: str, field: str):
    """
    驗證 .bin 檔案

    讀取指定股票的欄位，檢查資料完整性。
    """
    bin_path = QLIB_OUTPUT_DIR / "features" / stock_id / f"{field}.day.bin"

    if not bin_path.exists():
        return ValidateResponse(
            stock_id=stock_id,
            field=field,
            file_exists=False,
            record_count=0,
            nan_count=0,
        )

    # 讀取 .bin 檔
    data = np.fromfile(bin_path, dtype="<f")
    nan_mask = np.isnan(data)

    # 讀取 calendar 以對應日期
    calendar_path = QLIB_OUTPUT_DIR / "calendars" / "day.txt"
    dates = []
    if calendar_path.exists():
        with open(calendar_path) as f:
            dates = [line.strip() for line in f]

    # 取樣資料（最後 10 筆非 NaN）
    valid_indices = np.where(~nan_mask)[0]
    sample_indices = valid_indices[-10:] if len(valid_indices) > 0 else []
    sample_data = [
        {"date": dates[i] if i < len(dates) else str(i), "value": float(data[i])}
        for i in sample_indices
    ]

    return ValidateResponse(
        stock_id=stock_id,
        field=field,
        file_exists=True,
        record_count=len(data),
        nan_count=int(nan_mask.sum()),
        sample_data=sample_data,
    )


@router.get("/fields", response_model=FieldsResponse)
def list_fields(session: Session = Depends(get_db)):
    """列出所有可導出的欄位"""
    exporter = QlibExporter(session)
    fields = exporter.get_available_fields()

    return FieldsResponse(
        fields=[FieldInfo(**f) for f in fields],
        total=len(fields),
    )


@router.get("/status", response_model=StatusResponse)
def export_status():
    """檢查導出目錄狀態"""
    if not QLIB_OUTPUT_DIR.exists():
        return StatusResponse(
            exists=False,
            stocks=0,
            calendar_days=0,
            output_path=str(QLIB_OUTPUT_DIR),
        )

    features_dir = QLIB_OUTPUT_DIR / "features"
    stocks = list(features_dir.iterdir()) if features_dir.exists() else []

    calendar_path = QLIB_OUTPUT_DIR / "calendars" / "day.txt"
    calendar_days = 0
    if calendar_path.exists():
        with open(calendar_path) as f:
            calendar_days = sum(1 for _ in f)

    return StatusResponse(
        exists=True,
        stocks=len(stocks),
        calendar_days=calendar_days,
        output_path=str(QLIB_OUTPUT_DIR),
    )
