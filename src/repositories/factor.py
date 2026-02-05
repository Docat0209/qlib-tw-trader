from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.repositories.factors import ALL_FACTORS
from src.repositories.models import Factor, TrainingFactorResult

# 預設因子清單（從 factors 模組匯入）
# 包含：Alpha158 (~109) + 台股籌碼 (~130) + 交互因子 (~55) = ~294 個
DEFAULT_FACTORS = ALL_FACTORS


def seed_factors(session: Session, force: bool = False) -> int:
    """
    插入預設因子

    Args:
        session: 資料庫 Session
        force: 是否強制重新插入（會先清空現有因子）

    Returns:
        插入的因子數量
    """
    repo = FactorRepository(session)

    # 若不強制且已有資料，跳過
    existing = repo.get_all()
    if not force and existing:
        return 0

    # 強制模式：先刪除所有現有因子
    if force:
        for factor in existing:
            repo.delete(factor.id)

    # 插入預設因子
    count = 0
    for factor_data in DEFAULT_FACTORS:
        # 檢查是否已存在
        if repo.get_by_name(factor_data["name"]):
            continue
        repo.create(
            name=factor_data["name"],
            display_name=factor_data.get("display_name"),
            category=factor_data.get("category", "technical"),
            expression=factor_data["expression"],
            description=factor_data.get("description"),
        )
        count += 1

    return count


class FactorRepository:
    """因子定義存取"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        name: str,
        expression: str,
        display_name: str | None = None,
        category: str = "technical",
        description: str | None = None,
    ) -> Factor:
        """建立因子"""
        factor = Factor(
            name=name,
            display_name=display_name,
            category=category,
            expression=expression,
            description=description,
        )
        self._session.add(factor)
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def get_by_id(self, factor_id: int) -> Factor | None:
        """依 ID 取得因子"""
        stmt = select(Factor).where(Factor.id == factor_id)
        return self._session.execute(stmt).scalar()

    def get_by_name(self, name: str) -> Factor | None:
        """依名稱取得因子"""
        stmt = select(Factor).where(Factor.name == name)
        return self._session.execute(stmt).scalar()

    def get_enabled(self) -> list[Factor]:
        """取得所有啟用的因子"""
        stmt = select(Factor).where(Factor.enabled == True)
        return list(self._session.execute(stmt).scalars().all())

    def get_all(self, category: str | None = None, enabled: bool | None = None) -> list[Factor]:
        """取得所有因子（可篩選）"""
        stmt = select(Factor)
        if category is not None:
            stmt = stmt.where(Factor.category == category)
        if enabled is not None:
            stmt = stmt.where(Factor.enabled == enabled)
        return list(self._session.execute(stmt).scalars().all())

    def update(
        self,
        factor_id: int,
        name: str | None = None,
        display_name: str | None = None,
        category: str | None = None,
        expression: str | None = None,
        description: str | None = None,
    ) -> Factor | None:
        """更新因子"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return None
        if name is not None:
            factor.name = name
        if display_name is not None:
            factor.display_name = display_name
        if category is not None:
            factor.category = category
        if expression is not None:
            factor.expression = expression
        if description is not None:
            factor.description = description
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def delete(self, factor_id: int) -> bool:
        """刪除因子"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return False
        self._session.delete(factor)
        self._session.commit()
        return True

    def toggle(self, factor_id: int) -> Factor | None:
        """切換因子啟用狀態"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return None
        factor.enabled = not factor.enabled
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def set_enabled(self, factor_id: int, enabled: bool) -> Factor | None:
        """設定因子啟用狀態"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return None
        factor.enabled = enabled
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def get_selection_stats(self, factor_id: int) -> dict:
        """取得因子入選統計"""
        # 計算 times_evaluated（參與過的訓練次數）
        evaluated_stmt = (
            select(func.count())
            .select_from(TrainingFactorResult)
            .where(TrainingFactorResult.factor_id == factor_id)
        )
        times_evaluated = self._session.execute(evaluated_stmt).scalar() or 0

        # 計算 times_selected（被選中的次數）
        selected_stmt = (
            select(func.count())
            .select_from(TrainingFactorResult)
            .where(
                TrainingFactorResult.factor_id == factor_id,
                TrainingFactorResult.selected == True,
            )
        )
        times_selected = self._session.execute(selected_stmt).scalar() or 0

        selection_rate = times_selected / times_evaluated if times_evaluated > 0 else 0.0

        return {
            "times_evaluated": times_evaluated,
            "times_selected": times_selected,
            "selection_rate": selection_rate,
        }

    def get_selection_history(self, factor_id: int) -> list[dict]:
        """取得因子入選歷史"""
        from src.repositories.models import TrainingRun

        stmt = (
            select(TrainingFactorResult, TrainingRun)
            .join(TrainingRun)
            .where(TrainingFactorResult.factor_id == factor_id)
            .order_by(TrainingRun.started_at.desc())
        )
        results = self._session.execute(stmt).all()
        return [
            {
                "model_id": f"m{r.TrainingRun.id:03d}",
                "trained_at": r.TrainingRun.started_at.date().isoformat(),
                "selected": r.TrainingFactorResult.selected,
            }
            for r in results
        ]
