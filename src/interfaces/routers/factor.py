"""
因子管理 API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.factor import (
    FactorCreate,
    FactorDetailResponse,
    FactorListResponse,
    FactorResponse,
    FactorUpdate,
    SelectionHistory,
)
from src.repositories.factor import FactorRepository

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

    stats = repo.get_selection_stats(factor_id)
    return _factor_to_response(factor, stats)
