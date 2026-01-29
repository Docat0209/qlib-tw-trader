from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Factor


class FactorRepository:
    """因子定義存取"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        name: str,
        expression: str,
        description: str | None = None,
    ) -> Factor:
        """建立因子"""
        factor = Factor(
            name=name,
            expression=expression,
            description=description,
        )
        self._session.add(factor)
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def get_by_name(self, name: str) -> Factor | None:
        """依名稱取得因子"""
        stmt = select(Factor).where(Factor.name == name)
        return self._session.execute(stmt).scalar()

    def get_active(self) -> list[Factor]:
        """取得所有未排除的因子"""
        stmt = select(Factor).where(Factor.excluded == False)
        return list(self._session.execute(stmt).scalars().all())

    def get_all(self) -> list[Factor]:
        """取得所有因子"""
        stmt = select(Factor)
        return list(self._session.execute(stmt).scalars().all())

    def set_excluded(self, factor_id: int, excluded: bool) -> None:
        """設定因子排除狀態"""
        stmt = select(Factor).where(Factor.id == factor_id)
        factor = self._session.execute(stmt).scalar()
        if factor:
            factor.excluded = excluded
            self._session.commit()
