"""
Bootstrap 穩定性過濾器（輕量版）

透過多次子抽樣計算單因子 IC，只保留在 ≥75% 次抽樣中 IC 達標的因子。
這可以過濾掉因隨機變異而表現好的不穩定因子。

理論基礎：
- Meinshausen & Bühlmann (2010): Stability Selection
- 只有在多次抽樣中都表現穩定的因子才是真正可靠的

設計決策：
- 使用單因子 IC 作為選擇標準（輕量，避免計算量爆炸）
- 不使用 CPCV 作為 base_selector（30 × 15 = 450 次 fold 太慢）

參數（來自 constants.py）：
- BOOTSTRAP_N_ITERATIONS = 30
- BOOTSTRAP_STABILITY_THRESHOLD = 0.75
- BOOTSTRAP_SAMPLE_RATIO = 0.8
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.repositories.models import Factor
from src.shared.constants import (
    BOOTSTRAP_N_ITERATIONS,
    BOOTSTRAP_SAMPLE_RATIO,
    BOOTSTRAP_STABILITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 單因子 IC 門檻（低門檻，只排除完全無效的因子）
MIN_SINGLE_FACTOR_IC = 0.02


@dataclass
class BootstrapResult:
    """Bootstrap 過濾結果"""

    stable_factors: list[Factor]
    stable_factor_names: list[str]
    stability_scores: dict[str, float]  # factor_name -> stability score
    n_iterations: int
    threshold: float


class BootstrapStabilityFilter:
    """
    輕量版 Bootstrap 穩定性過濾器

    使用單因子 IC 作為選擇標準，透過多次子抽樣驗證因子的穩定性。
    只有在足夠比例的抽樣中 IC 都達標的因子才會被保留。
    """

    def __init__(
        self,
        n_iterations: int = BOOTSTRAP_N_ITERATIONS,
        stability_threshold: float = BOOTSTRAP_STABILITY_THRESHOLD,
        sample_ratio: float = BOOTSTRAP_SAMPLE_RATIO,
        min_ic: float = MIN_SINGLE_FACTOR_IC,
        random_state: int | None = 42,
    ):
        """
        Args:
            n_iterations: Bootstrap 迭代次數
            stability_threshold: 穩定性閾值（0-1，必須在多少比例的抽樣中達標）
            sample_ratio: 每次抽樣的比例
            min_ic: 單因子 IC 最低門檻
            random_state: 隨機種子
        """
        self.n_iterations = n_iterations
        self.stability_threshold = stability_threshold
        self.sample_ratio = sample_ratio
        self.min_ic = min_ic
        self.random_state = random_state

    def filter(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BootstrapResult:
        """
        執行 Bootstrap 穩定性過濾（輕量版）

        使用單因子 IC 作為選擇標準，避免計算量爆炸。

        Args:
            factors: 候選因子列表
            X: 特徵資料 (samples × factors)
            y: 標籤

        Returns:
            BootstrapResult
        """
        rng = np.random.default_rng(self.random_state)

        # 建立因子名稱 -> Factor 的映射
        factor_map = {f.name: f for f in factors}
        factor_names = list(factor_map.keys())

        # 初始化選擇計數
        selection_counts: dict[str, int] = defaultdict(int)
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_ratio)

        logger.info(
            f"Starting lightweight bootstrap filter: "
            f"{self.n_iterations} iterations, "
            f"{sample_size}/{n_samples} samples, "
            f"min_ic={self.min_ic}"
        )

        for i in range(self.n_iterations):
            # 隨機抽樣（不放回）
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]

            # 計算每個因子的單因子 IC
            for factor_name in factor_names:
                if factor_name not in X_sample.columns:
                    continue

                factor_values = X_sample[factor_name]

                # 跳過全為 NaN 的因子
                valid_mask = ~(factor_values.isna() | y_sample.isna())
                if valid_mask.sum() < 30:  # 至少需要 30 個有效樣本
                    continue

                # 計算 Spearman IC
                ic = factor_values[valid_mask].corr(y_sample[valid_mask], method="spearman")

                # 如果 IC 達標，計數 +1
                if not np.isnan(ic) and abs(ic) >= self.min_ic:
                    selection_counts[factor_name] += 1

            if (i + 1) % 10 == 0:
                logger.debug(f"Bootstrap progress: {i+1}/{self.n_iterations}")

        # 計算穩定性分數
        stability_scores = {
            name: count / self.n_iterations for name, count in selection_counts.items()
        }

        # 補充沒有被選中過的因子（穩定性分數為 0）
        for name in factor_names:
            if name not in stability_scores:
                stability_scores[name] = 0.0

        # 過濾穩定因子
        stable_factor_names = [
            name
            for name, score in stability_scores.items()
            if score >= self.stability_threshold
        ]
        stable_factors = [factor_map[name] for name in stable_factor_names if name in factor_map]

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
            stable_factor_names=stable_factor_names,
            stability_scores=stability_scores,
            n_iterations=self.n_iterations,
            threshold=self.stability_threshold,
        )
