"""
Bootstrap 穩定性過濾器

透過多次子抽樣執行因子選擇，只保留在 ≥75% 次抽樣中被選中的因子。
這可以過濾掉因隨機變異而被選中的不穩定因子。

理論基礎：
- Meinshausen & Bühlmann (2010): Stability Selection
- 只有在多次抽樣中都被選中的因子才是真正穩定的

參數（來自 constants.py）：
- BOOTSTRAP_N_ITERATIONS = 30（精簡版，平衡穩定性與計算成本）
- BOOTSTRAP_STABILITY_THRESHOLD = 0.75（≥75% 被選中才保留）
- BOOTSTRAP_SAMPLE_RATIO = 0.8（每次抽樣比例）
"""

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.repositories.models import Factor
from src.shared.constants import (
    BOOTSTRAP_N_ITERATIONS,
    BOOTSTRAP_SAMPLE_RATIO,
    BOOTSTRAP_STABILITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class FactorSelector(Protocol):
    """因子選擇器介面"""

    def select(
        self, factors: list[Factor], X: pd.DataFrame, y: pd.Series
    ) -> "SelectionResult":
        ...


@dataclass
class SelectionResult:
    """選擇結果"""

    selected_factors: list[Factor]


@dataclass
class BootstrapResult:
    """Bootstrap 過濾結果"""

    stable_factors: list[Factor]
    stability_scores: dict[str, float]  # factor_name -> stability score
    n_iterations: int
    threshold: float


class BootstrapStabilityFilter:
    """
    Bootstrap 穩定性過濾器

    透過多次子抽樣驗證因子選擇的穩定性。
    只有在足夠比例的抽樣中都被選中的因子才會被保留。
    """

    def __init__(
        self,
        n_iterations: int = BOOTSTRAP_N_ITERATIONS,
        stability_threshold: float = BOOTSTRAP_STABILITY_THRESHOLD,
        sample_ratio: float = BOOTSTRAP_SAMPLE_RATIO,
        random_state: int | None = 42,
    ):
        """
        Args:
            n_iterations: Bootstrap 迭代次數
            stability_threshold: 穩定性閾值（0-1，必須在多少比例的抽樣中被選中）
            sample_ratio: 每次抽樣的比例
            random_state: 隨機種子
        """
        self.n_iterations = n_iterations
        self.stability_threshold = stability_threshold
        self.sample_ratio = sample_ratio
        self.random_state = random_state

    def filter(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
        base_selector: FactorSelector,
    ) -> BootstrapResult:
        """
        執行 Bootstrap 穩定性過濾

        Args:
            factors: 候選因子列表
            X: 特徵資料 (samples × factors)
            y: 標籤
            base_selector: 基礎因子選擇器

        Returns:
            BootstrapResult
        """
        rng = np.random.default_rng(self.random_state)

        # 初始化選擇計數
        selection_counts: dict[str, int] = {f.name: 0 for f in factors}
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_ratio)

        logger.info(
            f"Starting bootstrap stability filter: "
            f"{self.n_iterations} iterations, "
            f"{sample_size}/{n_samples} samples per iteration"
        )

        for i in range(self.n_iterations):
            # 隨機抽樣（不放回）
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]

            # 執行因子選擇
            try:
                result = base_selector.select(factors, X_sample, y_sample)

                # 記錄被選中的因子
                for factor in result.selected_factors:
                    if factor.name in selection_counts:
                        selection_counts[factor.name] += 1

            except Exception as e:
                logger.warning(f"Bootstrap iteration {i+1} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.debug(f"Bootstrap progress: {i+1}/{self.n_iterations}")

        # 計算穩定性分數
        stability_scores = {
            name: count / self.n_iterations
            for name, count in selection_counts.items()
        }

        # 過濾穩定因子
        stable_factor_names = {
            name
            for name, score in stability_scores.items()
            if score >= self.stability_threshold
        }
        stable_factors = [f for f in factors if f.name in stable_factor_names]

        logger.info(
            f"Bootstrap filter complete: "
            f"{len(stable_factors)}/{len(factors)} factors passed "
            f"(threshold: {self.stability_threshold:.0%})"
        )

        # 記錄最穩定的因子
        top_factors = sorted(stability_scores.items(), key=lambda x: -x[1])[:10]
        logger.info(f"Top 10 stable factors: {top_factors}")

        return BootstrapResult(
            stable_factors=stable_factors,
            stability_scores=stability_scores,
            n_iterations=self.n_iterations,
            threshold=self.stability_threshold,
        )

    def filter_by_scores(
        self,
        factors: list[Factor],
        stability_scores: dict[str, float],
    ) -> list[Factor]:
        """
        根據已計算的穩定性分數過濾因子

        Args:
            factors: 候選因子列表
            stability_scores: 預先計算的穩定性分數

        Returns:
            通過閾值的因子列表
        """
        stable_factor_names = {
            name
            for name, score in stability_scores.items()
            if score >= self.stability_threshold
        }
        return [f for f in factors if f.name in stable_factor_names]
