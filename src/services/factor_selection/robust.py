"""穩健因子選擇

使用 Bootstrap 穩定性過濾 + CPCV 進行因子選擇。

流程：
1. Bootstrap 穩定性過濾：使用輕量單因子 IC，過濾不穩定因子
2. CPCV：對穩定因子進行嚴格的多路徑交叉驗證

參考文獻：
- Meinshausen & Bühlmann (2010). "Stability Selection"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
"""

import logging
from datetime import date
from typing import Any, Callable

import lightgbm as lgb
import pandas as pd

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector
from src.services.factor_selection.cpcv import CPCVSelector
from src.services.stability.bootstrap_filter import BootstrapStabilityFilter

logger = logging.getLogger(__name__)


class RobustFactorSelector(FactorSelector):
    """
    穩健因子選擇器

    使用 Bootstrap 穩定性過濾 + CPCV 進行因子選擇。
    """

    def __init__(
        self,
        # Bootstrap 參數
        enable_bootstrap: bool = True,
        bootstrap_n_iterations: int = 30,
        bootstrap_stability_threshold: float = 0.75,
        bootstrap_sample_ratio: float = 0.8,
        bootstrap_min_ic: float = 0.02,
        # CPCV 參數 (López de Prado, 2018; Harvey et al., 2016)
        cpcv_n_folds: int = 6,
        cpcv_n_test_folds: int = 2,
        cpcv_purge_days: int = 5,
        cpcv_embargo_days: int = 5,
        cpcv_significance: float = 0.05,  # 用於動態 t 閾值計算
        cpcv_min_positive_ratio: float = 0.6,  # 穩定性要求
        # LightGBM 參數
        lgbm_params: dict[str, Any] | None = None,
    ):
        """
        初始化選擇器

        Args:
            enable_bootstrap: 是否啟用 Bootstrap 前置過濾
            bootstrap_*: Bootstrap 參數
            cpcv_*: CPCV 參數 (基於 López de Prado 和 Harvey 等人的論文)
            lgbm_params: LightGBM 參數

        Note:
            t 閾值不再硬編碼，而是由 CPCVSelector 根據因子數量和路徑數量動態計算。
        """
        self.enable_bootstrap = enable_bootstrap

        # Bootstrap 過濾器
        self.bootstrap_filter = BootstrapStabilityFilter(
            n_iterations=bootstrap_n_iterations,
            stability_threshold=bootstrap_stability_threshold,
            sample_ratio=bootstrap_sample_ratio,
            min_ic=bootstrap_min_ic,
        )

        # CPCV 選擇器（t 閾值動態計算）
        self.cpcv_selector = CPCVSelector(
            n_folds=cpcv_n_folds,
            n_test_folds=cpcv_n_test_folds,
            purge_days=cpcv_purge_days,
            embargo_days=cpcv_embargo_days,
            significance_level=cpcv_significance,
            min_positive_ratio=cpcv_min_positive_ratio,
            lgbm_params=lgbm_params,
        )

        self.lgbm_params = lgbm_params or self._get_default_lgbm_params()

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

        流程：
        1. Bootstrap 穩定性過濾（如果啟用）
        2. CPCV 多路徑交叉驗證

        Args:
            factors: 候選因子列表
            X: 特徵資料
            y: 標籤資料
            on_progress: 進度回調

        Returns:
            選擇結果
        """
        stage_results = {}
        bootstrap_stats = {}
        factors_for_cpcv = factors

        # Step 1: Bootstrap 穩定性過濾
        if self.enable_bootstrap:
            if on_progress:
                on_progress(0, f"Bootstrap: Starting with {len(factors)} factors")

            bootstrap_result = self.bootstrap_filter.filter(factors, X, y)

            bootstrap_stats = {
                "bootstrap_input_factors": len(factors),
                "bootstrap_stable_factors": len(bootstrap_result.stable_factors),
                "bootstrap_threshold": bootstrap_result.threshold,
                "bootstrap_iterations": bootstrap_result.n_iterations,
            }
            stage_results["bootstrap"] = {
                "stable_factor_names": bootstrap_result.stable_factor_names,
                "top_stability_scores": dict(
                    sorted(bootstrap_result.stability_scores.items(), key=lambda x: -x[1])[:20]
                ),
            }

            factors_for_cpcv = bootstrap_result.stable_factors

            logger.info(
                f"Bootstrap filtered: {len(factors)} → {len(factors_for_cpcv)} factors"
            )

            if on_progress:
                on_progress(30, f"Bootstrap: {len(factors_for_cpcv)} stable factors")

        # Step 2: CPCV（如果有因子通過 Bootstrap）
        if len(factors_for_cpcv) == 0:
            logger.warning("No factors passed Bootstrap filter, returning empty result")
            return FactorSelectionResult(
                selected_factors=[],
                selection_stats={
                    "initial_factors": len(factors),
                    "final_factors": 0,
                    **bootstrap_stats,
                },
                method="robust",
                stage_results=stage_results,
            )

        if on_progress:
            progress_start = 30 if self.enable_bootstrap else 0
            on_progress(progress_start, f"CPCV: Starting with {len(factors_for_cpcv)} factors")

        # 調整 CPCV 進度回調的範圍
        def cpcv_progress(pct: float, msg: str) -> None:
            if on_progress:
                if self.enable_bootstrap:
                    # Bootstrap 用了 0-30%，CPCV 用 30-100%
                    adjusted_pct = 30 + pct * 0.7
                else:
                    adjusted_pct = pct
                on_progress(adjusted_pct, msg)

        cpcv_result = self.cpcv_selector.select(
            factors_for_cpcv, X, y, on_progress=cpcv_progress
        )
        stage_results["cpcv"] = cpcv_result.stage_results

        return FactorSelectionResult(
            selected_factors=cpcv_result.selected_factors,
            selection_stats={
                "initial_factors": len(factors),
                "final_factors": len(cpcv_result.selected_factors),
                **bootstrap_stats,
                **cpcv_result.selection_stats,
            },
            method="robust",
            stage_results=stage_results,
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
