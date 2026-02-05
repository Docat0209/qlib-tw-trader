"""
因子管理 API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.factor import (
    AvailableFieldsResponse,
    DeduplicateResponse,
    FactorCreate,
    FactorDetailResponse,
    FactorListResponse,
    FactorResponse,
    FactorUpdate,
    SeedResponse,
    SelectionHistory,
    ValidateRequest,
    ValidateResponse,
)
from src.repositories.factor import FactorRepository, seed_factors
from src.services.factor_validator import FactorValidator
from src.services.job_manager import broadcast_data_updated

router = APIRouter()


def _factor_to_response(factor, stats: dict) -> FactorResponse:
    """轉換 Factor Model 為 Response"""
    return FactorResponse(
        id=f"f{factor.id:03d}",
        name=factor.name,
        display_name=factor.display_name,
        category=factor.category,
        description=factor.description,
        formula=factor.expression,
        selection_rate=stats["selection_rate"],
        times_selected=stats["times_selected"],
        times_evaluated=stats["times_evaluated"],
        enabled=factor.enabled,
        created_at=factor.created_at,
    )


@router.get("", response_model=FactorListResponse)
async def list_factors(
    category: str | None = Query(None, description="篩選類別"),
    enabled: bool | None = Query(None, description="篩選啟用狀態"),
    session: Session = Depends(get_db),
):
    """取得因子清單"""
    repo = FactorRepository(session)
    factors = repo.get_all(category=category, enabled=enabled)

    items = []
    for factor in factors:
        stats = repo.get_selection_stats(factor.id)
        items.append(_factor_to_response(factor, stats))

    return FactorListResponse(items=items, total=len(items))


@router.get("/{factor_id}", response_model=FactorDetailResponse)
async def get_factor(
    factor_id: int,
    session: Session = Depends(get_db),
):
    """取得單一因子詳情"""
    repo = FactorRepository(session)
    factor = repo.get_by_id(factor_id)

    if not factor:
        raise HTTPException(status_code=404, detail="Factor not found")

    stats = repo.get_selection_stats(factor_id)
    history = repo.get_selection_history(factor_id)

    return FactorDetailResponse(
        id=f"f{factor.id:03d}",
        name=factor.name,
        display_name=factor.display_name,
        category=factor.category,
        description=factor.description,
        formula=factor.expression,
        selection_rate=stats["selection_rate"],
        times_selected=stats["times_selected"],
        times_evaluated=stats["times_evaluated"],
        enabled=factor.enabled,
        created_at=factor.created_at,
        selection_history=[SelectionHistory(**h) for h in history],
    )


@router.post("", response_model=FactorResponse, status_code=201)
async def create_factor(
    data: FactorCreate,
    session: Session = Depends(get_db),
):
    """新增因子"""
    repo = FactorRepository(session)

    # 檢查名稱是否重複
    if repo.get_by_name(data.name):
        raise HTTPException(status_code=400, detail="Factor name already exists")

    factor = repo.create(
        name=data.name,
        display_name=data.display_name,
        category=data.category,
        expression=data.formula,
        description=data.description,
    )

    await broadcast_data_updated("factors", "create", factor.id)

    stats = repo.get_selection_stats(factor.id)
    return _factor_to_response(factor, stats)


@router.put("/{factor_id}", response_model=FactorResponse)
async def update_factor(
    factor_id: int,
    data: FactorUpdate,
    session: Session = Depends(get_db),
):
    """更新因子"""
    repo = FactorRepository(session)

    # 檢查名稱是否重複（排除自己）
    if data.name:
        existing = repo.get_by_name(data.name)
        if existing and existing.id != factor_id:
            raise HTTPException(status_code=400, detail="Factor name already exists")

    factor = repo.update(
        factor_id=factor_id,
        name=data.name,
        display_name=data.display_name,
        category=data.category,
        expression=data.formula,
        description=data.description,
    )

    if not factor:
        raise HTTPException(status_code=404, detail="Factor not found")

    await broadcast_data_updated("factors", "update", factor_id)

    stats = repo.get_selection_stats(factor_id)
    return _factor_to_response(factor, stats)


@router.delete("/{factor_id}", status_code=204)
async def delete_factor(
    factor_id: int,
    session: Session = Depends(get_db),
):
    """刪除因子"""
    repo = FactorRepository(session)

    if not repo.delete(factor_id):
        raise HTTPException(status_code=404, detail="Factor not found")

    await broadcast_data_updated("factors", "delete", factor_id)


@router.patch("/{factor_id}/toggle", response_model=FactorResponse)
async def toggle_factor(
    factor_id: int,
    session: Session = Depends(get_db),
):
    """切換因子啟用狀態"""
    repo = FactorRepository(session)
    factor = repo.toggle(factor_id)

    if not factor:
        raise HTTPException(status_code=404, detail="Factor not found")

    await broadcast_data_updated("factors", "update", factor_id)

    stats = repo.get_selection_stats(factor_id)
    return _factor_to_response(factor, stats)


@router.post("/validate", response_model=ValidateResponse)
async def validate_expression(data: ValidateRequest):
    """驗證因子表達式"""
    validator = FactorValidator()
    result = validator.validate(data.expression)
    return ValidateResponse(
        valid=result.valid,
        error=result.error,
        fields_used=result.fields_used,
        operators_used=result.operators_used,
        warnings=result.warnings,
    )


@router.post("/seed", response_model=SeedResponse)
async def seed_default_factors(
    force: bool = Query(False, description="強制重新插入（會先清空現有因子）"),
    session: Session = Depends(get_db),
):
    """插入預設因子"""
    inserted = seed_factors(session, force=force)
    if inserted == 0 and not force:
        return SeedResponse(
            success=True,
            inserted=0,
            message="Factors already exist. Use force=true to re-seed.",
        )

    await broadcast_data_updated("factors", "create")

    return SeedResponse(
        success=True,
        inserted=inserted,
        message=f"Inserted {inserted} default factors.",
    )


@router.get("/available", response_model=AvailableFieldsResponse)
async def get_available_fields():
    """取得可用欄位和運算符"""
    validator = FactorValidator()
    return AvailableFieldsResponse(
        fields=validator.get_available_fields(),
        operators=validator.get_available_operators(),
    )


@router.post("/dedup", response_model=DeduplicateResponse)
async def deduplicate_factors(
    threshold: float = Query(0.99, description="相關係數閾值 (0.95-0.99)"),
    session: Session = Depends(get_db),
):
    """
    一次性因子去重

    計算啟用因子之間的相關性，禁用高度相關的冗餘因子。
    這是一次性操作，執行後訓練時不再需要重新計算。

    閾值說明：
    - 0.99: RD-Agent 預設，只移除極度冗餘
    - 0.95: 更積極去重
    """
    from pathlib import Path

    import pandas as pd

    from src.services.factor_selection.ic_dedup import ICDeduplicator
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    repo = FactorRepository(session)
    enabled_factors = repo.get_enabled()

    if len(enabled_factors) < 2:
        return DeduplicateResponse(
            success=True,
            total_factors=len(enabled_factors),
            kept_factors=len(enabled_factors),
            disabled_factors=0,
            disabled_names=[],
            message="Not enough factors to deduplicate.",
        )

    # 確保 Qlib 資料存在
    qlib_dir = Path("data/qlib")
    cal_file = qlib_dir / "calendars" / "day.txt"

    if not cal_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Qlib data not found. Please export data first.",
        )

    # 初始化 Qlib
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    try:
        qlib.init(provider_uri=str(qlib_dir), region=REG_CN)
    except Exception:
        pass  # Already initialized

    # 載入因子資料（使用最近一年的資料計算相關性）
    with open(cal_file) as f:
        dates = [line.strip() for line in f if line.strip()]

    if len(dates) < 60:
        raise HTTPException(
            status_code=400,
            detail="Not enough data for deduplication. Need at least 60 days.",
        )

    # 使用最近 252 天（約一年）
    start_date = dates[max(0, len(dates) - 252)]
    end_date = dates[-1]

    # 載入資料
    factor_exprs = {f.name: f.expression for f in enabled_factors}
    fields = list(factor_exprs.values())
    names = list(factor_exprs.keys())

    instruments = D.instruments("all")
    data = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
    )

    # 重命名欄位
    data.columns = names

    # 計算 label（用於排序 IC）
    close = D.features(
        instruments=instruments,
        fields=["Ref($close, -2) / Ref($close, -1) - 1"],
        start_time=start_date,
        end_time=end_date,
    )
    close.columns = ["label"]

    # 合併
    merged = data.join(close, how="inner")
    X = merged[names]
    y = merged["label"]

    # 執行去重
    deduplicator = ICDeduplicator(correlation_threshold=threshold)
    kept_factors, stats = deduplicator.deduplicate(enabled_factors, X, y)

    # 禁用被移除的因子
    kept_names = {f.name for f in kept_factors}
    disabled_names = []

    for factor in enabled_factors:
        if factor.name not in kept_names:
            repo.set_enabled(factor.id, False)
            disabled_names.append(factor.name)

    await broadcast_data_updated("factors", "update")

    return DeduplicateResponse(
        success=True,
        total_factors=len(enabled_factors),
        kept_factors=len(kept_factors),
        disabled_factors=len(disabled_names),
        disabled_names=disabled_names,
        message=f"Disabled {len(disabled_names)} redundant factors (correlation >= {threshold}).",
    )
