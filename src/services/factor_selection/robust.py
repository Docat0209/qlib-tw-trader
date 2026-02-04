"""穩健因子選擇

使用 CPCV (Combinatorial Purged Cross-Validation) 進行因子選擇。

為什麼只用 CPCV？
1. LightGBM 是嵌入式方法，內建特徵選擇（feature_pre_filter, feature_fraction, EFB）
2. CPCV 內部已用 LightGBM 訓練，自動處理無用特徵
3. 額外的預篩選層會增加假陰性（漏選有效因子）

參考文獻：
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
- LightGBM Documentation: https://lightgbm.readthedocs.io/en/latest/Features.html
"""

import logging
from datetime import date
from typing import Any, Callable

import lightgbm as lgb
import pandas as pd

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector
from src.services.factor_selection.cpcv import CPCVSelector

logger = logging.getLogger(__name__)


class RobustFactorSelector(FactorSelector):
    """
    穩健因子選擇器

    直接使用 CPCV 進行因子選擇，不需要額外的預篩選。
    """

    def __init__(
        self,
        # CPCV 參數 (López de Prado, 2018; Harvey et al., 2016)
        cpcv_n_folds: int = 6,
        cpcv_n_test_folds: int = 2,
        cpcv_purge_days: int = 5,
        cpcv_embargo_days: int = 5,
        cpcv_significance: float = 0.05,  # BH-FDR 標準閾值
        cpcv_min_t_stat: float = 2.0,  # Harvey et al. 建議 3.0，我們用 2.0
        cpcv_min_positive_ratio: float = 0.6,  # 穩定性要求
        # LightGBM 參數
        lgbm_params: dict[str, Any] | None = None,
    ):
        """
        初始化選擇器

        Args:
            cpcv_*: CPCV 參數 (基於 López de Prado 和 Harvey 等人的論文)
            lgbm_params: LightGBM 參數
        """
        self.cpcv_selector = CPCVSelector(
            n_folds=cpcv_n_folds,
            n_test_folds=cpcv_n_test_folds,
            purge_days=cpcv_purge_days,
            embargo_days=cpcv_embargo_days,
            significance_level=cpcv_significance,
            min_t_statistic=cpcv_min_t_stat,
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

        Args:
            factors: 候選因子列表
            X: 特徵資料
            y: 標籤資料
            on_progress: 進度回調

        Returns:
            選擇結果
        """
        if on_progress:
            on_progress(0, f"CPCV: Starting with {len(factors)} factors")

        # 直接使用 CPCV
        result = self.cpcv_selector.select(factors, X, y, on_progress=on_progress)

        # 更新 method 名稱
        return FactorSelectionResult(
            selected_factors=result.selected_factors,
            selection_stats={
                "initial_factors": len(factors),
                "final_factors": len(result.selected_factors),
                **result.selection_stats,
            },
            method="robust",
            stage_results={"cpcv": result.stage_results},
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
