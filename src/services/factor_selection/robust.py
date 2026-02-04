"""因子選擇器

支援三種文獻支持的方法：
1. none: Qlib 標準流程，不做選擇
2. dedup: RD-Agent IC 去重複
3. cpcv: CPCV + permutation importance（備用）

參考文獻：
- RD-Agent (Microsoft Research, 2025): https://arxiv.org/html/2505.15155v2
- Qlib (Microsoft): https://github.com/microsoft/qlib
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

import logging
from datetime import date
from typing import Any, Callable

import lightgbm as lgb
import pandas as pd

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector
from src.services.factor_selection.cpcv import CPCVSelector
from src.services.factor_selection.ic_dedup import ICDeduplicator
from src.shared.constants import IC_DEDUP_THRESHOLD

logger = logging.getLogger(__name__)


class RobustFactorSelector(FactorSelector):
    """
    因子選擇器

    支援三種模式（都有文獻支持）：
    - "none": Qlib 標準，不做選擇，依賴 LightGBM 內建機制
    - "dedup": RD-Agent IC 去重複，移除高相關因子
    - "cpcv": CPCV + permutation importance（備用）
    """

    def __init__(
        self,
        # 選擇方法
        method: str = "dedup",  # "none" | "dedup" | "cpcv"
        # IC Deduplication 參數 (RD-Agent)
        dedup_threshold: float = IC_DEDUP_THRESHOLD,
        # CPCV 參數（備用）
        cpcv_n_folds: int = 6,
        cpcv_n_test_folds: int = 2,
        cpcv_purge_days: int = 5,
        cpcv_embargo_days: int = 5,
        cpcv_min_positive_ratio: float = 0.5,
        # LightGBM 參數
        lgbm_params: dict[str, Any] | None = None,
    ):
        """
        初始化選擇器

        Args:
            method: 選擇方法
                - "none": 不做選擇（Qlib 標準）
                - "dedup": IC 去重複（RD-Agent）
                - "cpcv": CPCV（備用）
            dedup_threshold: IC 去重複閾值（RD-Agent 使用 0.99）
            cpcv_*: CPCV 參數（僅 method="cpcv" 時使用）
            lgbm_params: LightGBM 參數
        """
        self.method = method
        self.dedup_threshold = dedup_threshold
        self.lgbm_params = lgbm_params or self._get_default_lgbm_params()

        # IC Deduplicator (RD-Agent)
        self.deduplicator = ICDeduplicator(correlation_threshold=dedup_threshold)

        # CPCV Selector（備用）
        self.cpcv_selector = CPCVSelector(
            n_folds=cpcv_n_folds,
            n_test_folds=cpcv_n_test_folds,
            purge_days=cpcv_purge_days,
            embargo_days=cpcv_embargo_days,
            min_positive_ratio=cpcv_min_positive_ratio,
            lgbm_params=lgbm_params,
        )

        logger.info(f"RobustFactorSelector initialized with method={method}")

    def _get_default_lgbm_params(self) -> dict[str, Any]:
        """取得預設 LightGBM 參數"""
        return {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "device": "gpu",
            "gpu_use_dp": False,
        }

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

        # 方法 2: RD-Agent IC 去重複
        if self.method == "dedup":
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

        # 方法 3: CPCV（備用）
        if self.method == "cpcv":
            logger.info(f"Factor selection: method=cpcv")
            return self.cpcv_selector.select(factors, X, y, on_progress=on_progress)

        # 未知方法
        logger.warning(f"Unknown method: {self.method}, falling back to 'none'")
        return FactorSelectionResult(
            selected_factors=factors,
            selection_stats={"method": "unknown", "fallback": "none"},
            method="none",
        )

    def select_with_validation_split(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> tuple[FactorSelectionResult, Any]:
        """
        使用指定的訓練/驗證分割進行因子選擇

        Args:
            factors: 候選因子
            X: 特徵資料
            y: 標籤資料
            train_start/end: 訓練期
            valid_start/end: 驗證期
            on_progress: 進度回調

        Returns:
            (選擇結果, 最終模型)
        """
        # 取得日期索引
        if isinstance(X.index, pd.MultiIndex):
            dates = X.index.get_level_values("datetime")
            if hasattr(dates[0], "date"):
                date_values = pd.Series([d.date() for d in dates], index=X.index)
            else:
                date_values = pd.Series(dates, index=X.index)
        else:
            date_values = pd.to_datetime(X.index).date

        # 分割訓練和驗證資料
        train_mask = (date_values >= train_start) & (date_values <= train_end)
        valid_mask = (date_values >= valid_start) & (date_values <= valid_end)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # 執行因子選擇（使用訓練資料）
        result = self.select(factors, X_train, y_train, on_progress=on_progress)

        # 訓練最終模型
        if len(result.selected_factors) > 0:
            factor_names = [f.name for f in result.selected_factors]
            X_train_selected = X_train[factor_names]
            X_valid_selected = X_valid[factor_names]

            # 處理缺失值
            train_valid = ~(X_train_selected.isna().any(axis=1) | y_train.isna())
            valid_valid = ~(X_valid_selected.isna().any(axis=1) | y_valid.isna())

            X_train_clean = X_train_selected[train_valid]
            y_train_clean = y_train[train_valid]
            X_valid_clean = X_valid_selected[valid_valid]
            y_valid_clean = y_valid[valid_valid]

            model = lgb.LGBMRegressor(**self.lgbm_params)
            model.fit(
                X_train_clean,
                y_train_clean,
                eval_set=[(X_valid_clean, y_valid_clean)],
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )
        else:
            model = None

        return result, model
