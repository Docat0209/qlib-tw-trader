"""因子選擇基礎類"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from src.repositories.models import Factor


@dataclass
class FactorSelectionResult:
    """因子選擇結果"""

    selected_factors: list[Factor]
    selection_stats: dict[str, Any]
    method: str

    # 各階段中間結果（用於除錯和記錄）
    stage_results: dict[str, Any] = field(default_factory=dict)


class FactorSelector(ABC):
    """因子選擇器抽象基類"""

    @abstractmethod
    def select(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> FactorSelectionResult:
        """
        執行因子選擇

        Args:
            factors: 候選因子列表
            X: 特徵資料（columns = 因子名稱）
            y: 標籤資料
            on_progress: 進度回調函數 (progress: 0-100, message: str)

        Returns:
            選擇結果
        """
        pass
