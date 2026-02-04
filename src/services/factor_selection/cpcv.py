"""CPCV (Combinatorial Purged Cross-Validation) 因子選擇

Stage 2: 多路徑驗證，解決單一驗證期過擬合問題

原理：
- 產生 C(n_folds, n_test_folds) 種 train/test 組合
- 每個組合訓練模型、計算測試集 IC
- 只保留在多數組合都穩定有效的因子

關鍵機制：
- Purging: 移除測試集前後的訓練樣本（避免 label 洩漏）
- Embargo: 移除測試集後一段時間的訓練樣本（避免序列相關）
- BH-FDR: Benjamini-Hochberg 多重檢驗校正（比 Bonferroni 更適合探索性分析）
- t-statistic 門檻: 確保效果大小具統計意義

參考文獻：
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate"
- Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector
from src.shared.constants import (
    CPCV_EMBARGO_DAYS,
    CPCV_FALLBACK_MAX_FACTORS,
    CPCV_FALLBACK_POSITIVE_RATIO,
    CPCV_MIN_FACTORS_BEFORE_FALLBACK,
    CPCV_MIN_POSITIVE_RATIO,
    CPCV_N_FOLDS,
    CPCV_N_TEST_FOLDS,
    CPCV_PURGE_DAYS,
    CPCV_SIGNIFICANCE_ALPHA,
    CPCV_TIME_DECAY_RATE,
)

logger = logging.getLogger(__name__)


def calculate_dynamic_t_threshold(
    n_factors: int, n_paths: int, alpha: float = 0.05
) -> float:
    """
    計算動態 t 閾值

    基於 Bonferroni 校正 + 小樣本 t 分佈。

    原理：
    - Harvey et al. (2016) 的 t >= 3.0 假設大樣本 (df → ∞) 和 ~300 個因子
    - 我們需要根據實際的因子數量和路徑數量動態調整
    - 小樣本 (n_paths=15) 下，t 分佈的尾部更厚，臨界值更大

    Args:
        n_factors: 測試的因子數量（用於多重比較校正）
        n_paths: CPCV 路徑數量（決定自由度）
        alpha: 顯著水準（預設 0.05）

    Returns:
        動態計算的 t 閾值

    References:
        - Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
        - Bonferroni correction: α_adjusted = α / n_factors
        - t-distribution critical value with df = n_paths - 1
    """
    # Bonferroni 校正：調整顯著水準
    alpha_adjusted = alpha / max(1, n_factors)

    # 小樣本使用 t 分佈（雙尾檢驗）
    df = max(1, n_paths - 1)
    t_threshold = stats.t.ppf(1 - alpha_adjusted / 2, df)

    return float(t_threshold)


@dataclass
class CPCVFold:
    """CPCV 單一 fold 資訊"""

    fold_id: int
    test_fold_indices: tuple[int, ...]  # 哪些 fold 是測試集
    train_dates: list[date]
    test_dates: list[date]


class CPCVSelector(FactorSelector):
    """
    Combinatorial Purged Cross-Validation 因子選擇器

    在多條回測路徑上驗證因子穩定性
    """

    def __init__(
        self,
        n_folds: int = CPCV_N_FOLDS,
        n_test_folds: int = CPCV_N_TEST_FOLDS,
        purge_days: int = CPCV_PURGE_DAYS,
        embargo_days: int = CPCV_EMBARGO_DAYS,
        significance_level: float = CPCV_SIGNIFICANCE_ALPHA,  # BH-FDR + 動態 t 閾值
        min_positive_ratio: float = CPCV_MIN_POSITIVE_RATIO,  # 至少 60% 路徑有正向貢獻
        lgbm_params: dict[str, Any] | None = None,
    ):
        """
        初始化 CPCV 選擇器

        Args:
            n_folds: 時間區塊數（López de Prado 示例用 6）
            n_test_folds: 每次組合中的測試 fold 數（López de Prado 示例用 2）
            purge_days: Purging 天數（label lookahead 防護）
            embargo_days: Embargo 天數（序列相關防護）
            significance_level: 顯著水準，用於 BH-FDR 和動態 t 閾值計算
            min_positive_ratio: 最低正向路徑比例（穩定性要求）
            lgbm_params: LightGBM 參數

        Note:
            t 閾值不再硬編碼為 3.0，而是根據因子數量和路徑數量動態計算。
            這遵循 Harvey et al. (2016) 的 Bonferroni 校正原理，同時考慮
            小樣本的 t 分佈特性。
        """
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.significance_level = significance_level
        self.min_positive_ratio = min_positive_ratio
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

    def _generate_folds(self, dates: np.ndarray) -> list[CPCVFold]:
        """
        生成 CPCV fold 組合

        Args:
            dates: 排序後的唯一日期陣列

        Returns:
            C(n_folds, n_test_folds) 個 fold
        """
        n_dates = len(dates)
        fold_size = n_dates // self.n_folds

        # 將日期分成 n_folds 個區塊
        fold_boundaries = []
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else n_dates
            fold_boundaries.append((start_idx, end_idx))

        # 生成所有 C(n_folds, n_test_folds) 組合
        folds = []
        for fold_id, test_combo in enumerate(
            combinations(range(self.n_folds), self.n_test_folds)
        ):
            # 測試日期
            test_dates = []
            for fold_idx in test_combo:
                start_idx, end_idx = fold_boundaries[fold_idx]
                test_dates.extend(dates[start_idx:end_idx])

            # 訓練日期（非測試 + purging + embargo）
            test_date_set = set(test_dates)
            train_dates = []

            for fold_idx in range(self.n_folds):
                if fold_idx in test_combo:
                    continue  # 跳過測試 fold

                start_idx, end_idx = fold_boundaries[fold_idx]
                fold_dates = dates[start_idx:end_idx]

                # 應用 purging 和 embargo
                for d in fold_dates:
                    if self._should_include_in_train(d, test_dates, dates):
                        train_dates.append(d)

            folds.append(
                CPCVFold(
                    fold_id=fold_id,
                    test_fold_indices=test_combo,
                    train_dates=sorted(train_dates),
                    test_dates=sorted(test_dates),
                )
            )

        return folds

    def _should_include_in_train(
        self, train_date: date, test_dates: list[date], all_dates: np.ndarray
    ) -> bool:
        """
        判斷訓練日期是否應該包含（purging + embargo）

        Purging: 移除接近測試集的訓練樣本（label 可能洩漏）
        Embargo: 移除測試集結束後一段時間的訓練樣本（序列相關）
        """
        min_test = min(test_dates)
        max_test = max(test_dates)

        # Purging: 訓練日期不能太接近測試集開始
        # （因為 label 是未來收益，可能洩漏）
        purge_start = min_test - timedelta(days=self.purge_days)
        if train_date > purge_start and train_date < min_test:
            return False

        # Embargo: 測試集結束後一段時間不能用於訓練
        # （避免序列相關導致的資訊洩漏）
        embargo_end = max_test + timedelta(days=self.embargo_days)
        if train_date > max_test and train_date < embargo_end:
            return False

        return True

    def select(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> FactorSelectionResult:
        """
        執行 CPCV 因子選擇

        Args:
            factors: 候選因子列表
            X: 特徵資料
            y: 標籤資料
            on_progress: 進度回調

        Returns:
            通過多路徑驗證的因子
        """
        if on_progress:
            on_progress(0, f"CPCV: Preparing {len(factors)} factors...")

        # 建立因子名稱映射
        factor_map = {f.name: f for f in factors}
        factor_names = [f.name for f in factors]

        # 確保 X 只包含因子欄位
        X_subset = X[factor_names].copy()

        # 對齊 X 和 y
        common_idx = X_subset.index.intersection(y.index)
        X_aligned = X_subset.loc[common_idx]
        y_aligned = y.loc[common_idx]

        # 取得日期索引
        if isinstance(X_aligned.index, pd.MultiIndex):
            dates = X_aligned.index.get_level_values("datetime")
            if hasattr(dates[0], "date"):
                dates = pd.Series([d.date() for d in dates])
        else:
            dates = pd.to_datetime(X_aligned.index).date

        unique_dates = np.array(sorted(set(dates)))

        if len(unique_dates) < self.n_folds * 20:
            logger.warning(f"CPCV: Not enough dates ({len(unique_dates)})")
            return FactorSelectionResult(
                selected_factors=factors,
                selection_stats={"error": "not_enough_dates"},
                method="cpcv",
            )

        # 生成 CPCV fold
        folds = self._generate_folds(unique_dates)
        n_paths = len(folds)

        if on_progress:
            on_progress(5, f"CPCV: Generated {n_paths} paths")

        # 在每個 fold 上評估每個因子
        factor_ics: dict[str, list[float]] = defaultdict(list)

        for fold_idx, fold in enumerate(folds):
            if on_progress:
                progress = 5 + (fold_idx / n_paths) * 85
                on_progress(progress, f"CPCV: Path {fold_idx + 1}/{n_paths}")

            # 建立日期到 boolean mask 的映射
            train_date_set = set(fold.train_dates)
            test_date_set = set(fold.test_dates)

            # 建立 mask
            if isinstance(X_aligned.index, pd.MultiIndex):
                date_values = X_aligned.index.get_level_values("datetime")
                if hasattr(date_values[0], "date"):
                    date_values = pd.Series([d.date() for d in date_values], index=X_aligned.index)
                train_mask = date_values.isin(train_date_set)
                test_mask = date_values.isin(test_date_set)
            else:
                date_values = pd.to_datetime(X_aligned.index).date
                train_mask = pd.Series(date_values).isin(train_date_set).values
                test_mask = pd.Series(date_values).isin(test_date_set).values

            X_train = X_aligned[train_mask]
            X_test = X_aligned[test_mask]
            y_train = y_aligned[train_mask]
            y_test = y_aligned[test_mask]

            # 處理缺失值
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]

            if len(X_train) < 100 or len(X_test) < 20:
                logger.warning(f"CPCV: Fold {fold_idx} has insufficient data")
                continue

            # 訓練模型（使用所有因子）
            try:
                model = lgb.LGBMRegressor(**self.lgbm_params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(10, verbose=False)],
                )
            except Exception as e:
                logger.warning(f"CPCV: Model training failed for fold {fold_idx}: {e}")
                continue

            # 計算每個因子的貢獻（使用 permutation importance）
            # 使用 Spearman 相關係數以與其他 IC 計算保持一致
            base_pred = model.predict(X_test)
            base_ic, _ = stats.spearmanr(base_pred, y_test.values)

            for factor_name in factor_names:
                # 打亂單一因子
                X_permuted = X_test.copy()
                X_permuted[factor_name] = np.random.permutation(
                    X_permuted[factor_name].values
                )

                permuted_pred = model.predict(X_permuted)
                permuted_ic, _ = stats.spearmanr(permuted_pred, y_test.values)

                # 因子重要性 = IC 下降幅度
                importance = base_ic - permuted_ic
                factor_ics[factor_name].append(importance)

        if on_progress:
            on_progress(90, "CPCV: Statistical testing...")

        # 計算時間衰減權重（近期路徑權重更高）
        # 權重基於 fold 的最大測試區塊索引：測試越近期的路徑權重越高
        path_weights = []
        for fold in folds:
            max_test_idx = max(fold.test_fold_indices)
            # 近期（高索引）權重高，遠期（低索引）權重低
            weight = CPCV_TIME_DECAY_RATE ** (self.n_folds - 1 - max_test_idx)
            path_weights.append(weight)

        # 正規化權重
        total_weight = sum(path_weights)
        path_weights = [w / total_weight for w in path_weights]

        logger.info(
            f"CPCV: Time decay weights: min={min(path_weights):.3f}, "
            f"max={max(path_weights):.3f}, decay_rate={CPCV_TIME_DECAY_RATE}"
        )

        # 計算每個因子的統計量（使用時間加權）
        n_factors = len(factor_names)
        factor_stats = {}
        factor_p_values = {}  # 用於 BH-FDR 校正

        for factor_name in factor_names:
            ics = factor_ics[factor_name]
            if len(ics) < 3:
                logger.warning(f"CPCV: {factor_name} skipped (only {len(ics)} paths)")
                continue

            # 取得對應的權重（可能某些 fold 被跳過，長度不一定相等）
            # 使用已計算的路徑數量來對齊權重
            n_ics = len(ics)
            if n_ics == len(path_weights):
                weights = np.array(path_weights)
            else:
                # 如果長度不一致，使用均勻權重
                weights = np.ones(n_ics) / n_ics

            ics_array = np.array(ics)

            # 時間加權平均
            weighted_mean = np.average(ics_array, weights=weights)

            # 時間加權標準差
            weighted_var = np.average((ics_array - weighted_mean) ** 2, weights=weights)
            weighted_std = np.sqrt(weighted_var)

            # 有效樣本數（Kish's effective sample size）
            # n_eff = (Σw)² / Σw² = 1 / Σ(w_normalized)²
            effective_n = 1.0 / np.sum(weights ** 2)

            # 同時計算未加權統計量以供參考
            raw_mean = np.mean(ics_array)
            raw_std = np.std(ics_array, ddof=1)

            n_positive = sum(1 for ic in ics if ic > 0)
            positive_ratio = n_positive / len(ics)

            # 使用時間加權的 t-test
            if weighted_std > 0:
                t_stat = weighted_mean / (weighted_std / np.sqrt(effective_n))
                p_value = 1 - stats.t.cdf(t_stat, df=max(1, effective_n - 1))
            else:
                t_stat = 0
                p_value = 1.0

            factor_stats[factor_name] = {
                "mean_importance": float(weighted_mean),
                "std_importance": float(weighted_std),
                "raw_mean": float(raw_mean),
                "raw_std": float(raw_std),
                "effective_n": float(effective_n),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "n_paths": len(ics),
                "positive_ratio": float(positive_ratio),
            }
            factor_p_values[factor_name] = p_value

        # === 動態 t 閾值選擇標準 ===
        # 基於 Harvey et al. (2016) 的多重比較校正原理，但動態調整：
        # - 根據實際測試的因子數量計算 Bonferroni 校正
        # - 根據 CPCV 路徑數量（小樣本）使用 t 分佈而非正態分佈
        #
        # 同時使用 BH-FDR 作為備選路徑（對於邊緣情況更有統計功效）
        # 兩者都要求: 正向路徑比例 >= 門檻 (穩定性要求)

        # 動態計算 t 閾值
        dynamic_t_threshold = calculate_dynamic_t_threshold(
            n_factors=n_factors,
            n_paths=n_paths,
            alpha=self.significance_level,
        )

        # Fallback t 閾值：使用較寬鬆的校正（10% 顯著水準）
        fallback_t_threshold = calculate_dynamic_t_threshold(
            n_factors=n_factors,
            n_paths=n_paths,
            alpha=0.10,  # 較寬鬆的顯著水準
        )

        logger.info(
            f"CPCV: Dynamic t-thresholds calculated: "
            f"primary={dynamic_t_threshold:.3f} (α={self.significance_level}), "
            f"fallback={fallback_t_threshold:.3f} (α=0.10), "
            f"n_factors={n_factors}, n_paths={n_paths}"
        )

        sorted_factors = sorted(factor_p_values.items(), key=lambda x: x[1])
        n_tested = len(sorted_factors)

        # 找出 BH-FDR 校正後顯著的最大 rank
        bh_cutoff_rank = 0
        for rank, (factor_name, p_value) in enumerate(sorted_factors, 1):
            bh_threshold = (rank / n_tested) * self.significance_level
            if p_value <= bh_threshold:
                bh_cutoff_rank = rank
            factor_stats[factor_name]["bh_threshold"] = float(bh_threshold)

        # 選擇通過條件的因子
        selected_names = []
        for rank, (factor_name, p_value) in enumerate(sorted_factors, 1):
            stats_dict = factor_stats[factor_name]
            t_stat = stats_dict["t_stat"]
            positive_ratio = stats_dict["positive_ratio"]

            # 條件檢查（使用動態閾值）
            pass_dynamic_t = t_stat >= dynamic_t_threshold  # 動態 Bonferroni 校正
            pass_bh = rank <= bh_cutoff_rank  # BH-FDR 校正
            pass_stability = positive_ratio >= self.min_positive_ratio  # 穩定性

            stats_dict["pass_dynamic_t"] = pass_dynamic_t
            stats_dict["dynamic_t_threshold"] = dynamic_t_threshold
            stats_dict["pass_bh"] = pass_bh
            stats_dict["pass_stability"] = pass_stability

            # 選擇邏輯：
            # 路徑 A: t >= 動態閾值 (Bonferroni) + 穩定性 → 直接選擇
            # 路徑 B: BH-FDR + 穩定性 → 選擇
            if (pass_dynamic_t and pass_stability) or (pass_bh and pass_stability):
                selected_names.append(factor_name)
                stats_dict["selected"] = True
                stats_dict["selection_path"] = "bonferroni" if pass_dynamic_t else "bh_fdr"
            else:
                stats_dict["selected"] = False

        logger.info(
            f"CPCV: Primary selection: {len(selected_names)} factors "
            f"(Bonferroni path: {sum(1 for n in selected_names if factor_stats[n].get('selection_path') == 'bonferroni')}, "
            f"BH-FDR path: {sum(1 for n in selected_names if factor_stats[n].get('selection_path') == 'bh_fdr')})"
        )

        # 備用：如果主選擇的因子太少，使用較寬鬆的動態閾值補充
        # 原因：只有 1-2 個因子的模型太不穩定
        if len(selected_names) < CPCV_MIN_FACTORS_BEFORE_FALLBACK:
            logger.warning(
                f"CPCV: Only {len(selected_names)} factors passed primary criteria "
                f"(< {CPCV_MIN_FACTORS_BEFORE_FALLBACK}), "
                f"using fallback with relaxed threshold "
                f"(t>={fallback_t_threshold:.3f}, positive_ratio>={CPCV_FALLBACK_POSITIVE_RATIO})"
            )
            # 找出符合 fallback 條件但尚未被選中的因子
            already_selected = set(selected_names)
            relaxed_candidates = [
                (name, factor_stats[name]["t_stat"])
                for name in factor_p_values.keys()
                if name not in already_selected
                and factor_stats[name]["t_stat"] >= fallback_t_threshold
                and factor_stats[name]["positive_ratio"] >= CPCV_FALLBACK_POSITIVE_RATIO
            ]
            relaxed_candidates.sort(key=lambda x: x[1], reverse=True)

            # 補充到至少 CPCV_MIN_FACTORS_BEFORE_FALLBACK 個，但不超過 CPCV_FALLBACK_MAX_FACTORS
            need_count = CPCV_MIN_FACTORS_BEFORE_FALLBACK - len(selected_names)
            max_supplement = min(CPCV_FALLBACK_MAX_FACTORS - len(selected_names), need_count + 5)
            supplement_names = [name for name, _ in relaxed_candidates[:max_supplement]]

            for name in supplement_names:
                factor_stats[name]["fallback_selected"] = True
                factor_stats[name]["fallback_t_threshold"] = fallback_t_threshold
                selected_names.append(name)

            logger.info(
                f"CPCV: Fallback added {len(supplement_names)} factors, "
                f"total now {len(selected_names)} factors"
            )

        # 打印統計摘要
        all_t_stats = [s["t_stat"] for s in factor_stats.values()]
        all_p_values = [s["p_value"] for s in factor_stats.values()]
        if all_t_stats:
            logger.info(
                f"CPCV: t_stat range=[{min(all_t_stats):.3f}, {max(all_t_stats):.3f}], "
                f"BH cutoff rank={bh_cutoff_rank}/{n_tested}"
            )
            logger.info(f"CPCV: Selected {len(selected_names)} factors")

        # 轉換為 Factor 物件
        selected_factors = [factor_map[name] for name in selected_names if name in factor_map]

        if on_progress:
            on_progress(100, f"CPCV: Selected {len(selected_factors)} factors")

        return FactorSelectionResult(
            selected_factors=selected_factors,
            selection_stats={
                "total_factors": len(factors),
                "selected_count": len(selected_factors),
                "n_paths": n_paths,
                "purge_days": self.purge_days,
                "embargo_days": self.embargo_days,
                "significance_level": self.significance_level,
                "dynamic_t_threshold": dynamic_t_threshold,
                "fallback_t_threshold": fallback_t_threshold,
                "min_positive_ratio": self.min_positive_ratio,
                "correction_method": "dynamic-bonferroni + benjamini-hochberg",
                "bh_cutoff_rank": bh_cutoff_rank,
                "time_decay_rate": CPCV_TIME_DECAY_RATE,
            },
            method="cpcv",
            stage_results={
                "factor_stats": factor_stats,
                "selected_names": selected_names,
            },
        )
