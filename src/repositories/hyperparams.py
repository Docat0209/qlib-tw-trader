"""超參數組 Repository"""

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Hyperparams
from src.shared.constants import TZ_TAIPEI


class HyperparamsRepository:
    """超參數組存取"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        name: str,
        n_periods: int,
        params: dict,
        stability: dict,
        periods: list[dict],
    ) -> Hyperparams:
        """建立超參數組"""
        hp = Hyperparams(
            name=name,
            cultivated_at=datetime.now(TZ_TAIPEI),
            n_periods=n_periods,
            params_json=json.dumps(params),
            stability_json=json.dumps(stability),
            periods_json=json.dumps(periods),
        )
        self._session.add(hp)
        self._session.commit()
        self._session.refresh(hp)
        return hp

    def get_all(self) -> list[Hyperparams]:
        """取得所有超參數組（按培養時間降序）"""
        stmt = select(Hyperparams).order_by(Hyperparams.cultivated_at.desc())
        return list(self._session.execute(stmt).scalars().all())

    def get_latest(self) -> Hyperparams | None:
        """取得最新的超參數組"""
        stmt = select(Hyperparams).order_by(Hyperparams.cultivated_at.desc()).limit(1)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_id(self, hp_id: int) -> Hyperparams | None:
        """依 ID 取得"""
        return self._session.get(Hyperparams, hp_id)

    def get_by_name(self, name: str) -> Hyperparams | None:
        """依名稱取得"""
        stmt = select(Hyperparams).where(Hyperparams.name == name)
        return self._session.execute(stmt).scalar_one_or_none()

    def update_name(self, hp_id: int, name: str) -> Hyperparams | None:
        """更新名稱"""
        hp = self.get_by_id(hp_id)
        if hp:
            hp.name = name
            self._session.commit()
            self._session.refresh(hp)
        return hp

    def delete(self, hp_id: int) -> bool:
        """刪除超參數組"""
        hp = self.get_by_id(hp_id)
        if not hp:
            return False

        self._session.delete(hp)
        self._session.commit()
        return True

    def get_params(self, hp_id: int) -> dict | None:
        """取得超參數 dict"""
        hp = self.get_by_id(hp_id)
        if hp:
            return json.loads(hp.params_json)
        return None

    def get_stability(self, hp_id: int) -> dict | None:
        """取得穩定性指標 dict"""
        hp = self.get_by_id(hp_id)
        if hp:
            return json.loads(hp.stability_json)
        return None

    def get_periods(self, hp_id: int) -> list[dict] | None:
        """取得各窗口結果"""
        hp = self.get_by_id(hp_id)
        if hp:
            return json.loads(hp.periods_json)
        return None
