"""因子選擇器

支援兩種文獻支持的方法：
1. none: Qlib 標準流程，不做選擇
2. dedup: RD-Agent IC 去重複

參考文獻：
- RD-Agent (Microsoft Research, 2025): https://arxiv.org/html/2505.15155v2
- Qlib (Microsoft): https://github.com/microsoft/qlib
"""

import logging
from typing import Callable

import pandas as pd

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector
from src.services.factor_selection.ic_dedup import ICDeduplicator
from src.shared.constants import IC_DEDUP_THRESHOLD

logger = logging.getLogger(__name__)


class RobustFactorSelector(FactorSelector):
    """
    因子選擇器

    支援兩種模式（都有文獻支持）：
    - "none": Qlib 標準，不做選擇，依賴 LightGBM 內建機制
    - "dedup": RD-Agent IC 去重複，移除高相關因子
    """

    def __init__(
        self,
        method: str = "dedup",  # "none" | "dedup"
        dedup_threshold: float = IC_DEDUP_THRESHOLD,
    ):
        """
        初始化選擇器

        Args:
            method: 選擇方法
                - "none": 不做選擇（Qlib 標準）
                - "dedup": IC 去重複（RD-Agent）
            dedup_threshold: IC 去重複閾值（RD-Agent 使用 0.99）
        """
        self.method = method
        self.dedup_threshold = dedup_threshold
        self.deduplicator = ICDeduplicator(correlation_threshold=dedup_threshold)

        logger.info(f"RobustFactorSelector initialized with method={method}")

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
            X: 特徵資料
            y: 標籤資料
            on_progress: 進度回調

        Returns:
            選擇結果
        """
        if on_progress:
            on_progress(0, f"Factor selection: {len(factors)} factors, method={self.method}")

        # 方法 1: Qlib 標準（不做選擇）
        if self.method == "none":
            logger.info(f"Factor selection: method=none, keeping all {len(factors)} factors")
            if on_progress:
                on_progress(100, f"No selection: {len(factors)} factors")

            return FactorSelectionResult(
                selected_factors=factors,
                selection_stats={
                    "method": "none",
                    "input_count": len(factors),
                    "output_count": len(factors),
                },
                method="none",
            )

        # 方法 2: RD-Agent IC 去重複（預設）
        logger.info(f"Factor selection: method=dedup, threshold={self.dedup_threshold}")
        if on_progress:
            on_progress(10, f"IC Dedup: Processing {len(factors)} factors...")

        kept_factors, stats = self.deduplicator.deduplicate(factors, X, y)

        if on_progress:
            on_progress(100, f"IC Dedup: {len(kept_factors)} factors kept")

        return FactorSelectionResult(
            selected_factors=kept_factors,
            selection_stats=stats,
            method="dedup",
        )
