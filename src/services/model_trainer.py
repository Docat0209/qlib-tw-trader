"""
模型訓練服務 - LightGBM IC 增量選擇法
"""

import hashlib
import json
import math
import pickle
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor, TrainingRun
from src.repositories.training import TrainingRepository
from src.shared.constants import TRAIN_DAYS, VALID_DAYS

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# 模型檔案目錄
MODELS_DIR = Path("data/models")

# 超參數檔案路徑
HYPERPARAMS_FILE = Path("data/hyperparams.json")


@dataclass
class FactorEvalResult:
    """因子評估結果"""

    factor_id: int
    factor_name: str
    ic_value: float
    selected: bool


@dataclass
class TrainingResult:
    """訓練結果"""

    run_id: int
    model_name: str
    model_ic: float
    icir: float | None
    selected_factor_ids: list[int]
    all_results: list[FactorEvalResult]


@dataclass
class PeriodResult:
    """單一窗口的優化結果"""

    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    best_ic: float
    params: dict


@dataclass
class CultivationResult:
    """超參數培養結果"""

    cultivated_at: str
    n_periods: int
    params: dict
    stability: dict[str, float]
    periods: list[PeriodResult] = field(default_factory=list)


def scale_params_for_factor_count(
    base_params: dict,
    base_factor_count: int,
    actual_factor_count: int,
) -> dict:
    """
    根據實際因子數量縮放超參數

    原理：
    - 超參數是針對 base_factor_count 個因子優化的
    - 但實際使用時可能只有 actual_factor_count 個因子
    - 需要調整模型複雜度以避免過擬合

    縮放規則：
    - num_leaves: sqrt 縮放，下限 8
    - max_depth: 減少 log2(base/actual)，下限 3
    - min_data_in_leaf: sqrt 縮放，下限 10
    - lambda_l1/l2: sqrt 縮放（低維需較少正則化）
    - feature_fraction: 少因子時用 1.0
    """
    if actual_factor_count >= base_factor_count:
        return base_params.copy()

    ratio = actual_factor_count / base_factor_count
    sqrt_ratio = math.sqrt(ratio)

    scaled = base_params.copy()

    # num_leaves: 與因子數正相關
    if "num_leaves" in base_params:
        scaled["num_leaves"] = max(8, int(base_params["num_leaves"] * sqrt_ratio))

    # max_depth: 對數縮減
    if "max_depth" in base_params:
        reduction = int(math.log2(max(1, base_factor_count / actual_factor_count)))
        scaled["max_depth"] = max(3, base_params["max_depth"] - reduction)

    # min_data_in_leaf: 允許更小的葉節點
    if "min_data_in_leaf" in base_params:
        scaled["min_data_in_leaf"] = max(10, int(base_params["min_data_in_leaf"] * sqrt_ratio))

    # 正則化：低維需較少正則化
    if "lambda_l1" in base_params:
        scaled["lambda_l1"] = base_params["lambda_l1"] * sqrt_ratio
    if "lambda_l2" in base_params:
        scaled["lambda_l2"] = base_params["lambda_l2"] * sqrt_ratio

    # feature_fraction：少因子時使用全部
    if "feature_fraction" in base_params and actual_factor_count <= 5:
        scaled["feature_fraction"] = 1.0

    return scaled


def get_conservative_default_params(factor_count: int) -> dict:
    """
    根據因子數量返回保守的預設參數

    設計原則：寧可欠擬合也不過擬合
    """
    base = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
        "learning_rate": 0.05,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    if factor_count <= 3:
        base.update({
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": 15,
            "feature_fraction": 1.0,
            "lambda_l1": 2.0,
            "lambda_l2": 2.0,
        })
    elif factor_count <= 6:
        base.update({
            "num_leaves": 12,
            "max_depth": 4,
            "min_data_in_leaf": 18,
            "feature_fraction": 0.9,
            "lambda_l1": 3.0,
            "lambda_l2": 3.0,
        })
    elif factor_count <= 12:
        base.update({
            "num_leaves": 16,
            "max_depth": 4,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "lambda_l1": 5.0,
            "lambda_l2": 5.0,
        })
    else:
        base.update({
            "num_leaves": 24,
            "max_depth": 5,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "lambda_l1": 8.0,
            "lambda_l2": 8.0,
        })

    return base


class ModelTrainer:
    """模型訓練器 - LightGBM IC 增量選擇法 + Optuna 超參數優化"""

    # Optuna 調參設定
    OPTUNA_N_TRIALS = 50  # 搜索次數
    OPTUNA_TIMEOUT = 300  # 超時秒數（5分鐘）

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False
        self._last_ic_std: float | None = None  # 用於 ICIR 計算
        self._data_cache: dict[str, pd.DataFrame] = {}  # 資料快取
        self._optimized_params: dict | None = None  # Optuna 優化後的參數

    def _init_qlib(self, force: bool = False) -> None:
        """
        初始化 qlib

        Args:
            force: 強制重新初始化（用於導出新資料後）
        """
        if self._qlib_initialized and not force:
            return

        try:
            import qlib
            from qlib.config import REG_CN

            # 強制重新初始化：清除快取並重新載入
            qlib.init(
                provider_uri=str(self.data_dir),
                region=REG_CN,
            )
            self._qlib_initialized = True
            self._data_cache.clear()  # 清除資料快取
        except ImportError:
            raise RuntimeError("qlib is not installed. Please run: pip install pyqlib")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def _get_instruments(self) -> list[str]:
        """從 qlib instruments 目錄讀取股票清單"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        # 備選：從 features 目錄取得
        features_dir = self.data_dir / "features"
        if features_dir.exists():
            return [d.name for d in features_dir.iterdir() if d.is_dir()]

        return []

    def get_data_date_range(self) -> tuple[date | None, date | None]:
        """取得 qlib 資料的日期範圍"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if not instruments_file.exists():
            return None, None

        min_start = None
        max_end = None

        with open(instruments_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        start = date.fromisoformat(parts[1])
                        end = date.fromisoformat(parts[2])
                        if min_start is None or start < min_start:
                            min_start = start
                        if max_end is None or end > max_end:
                            max_end = end
                    except ValueError:
                        continue

        return min_start, max_end

    def _load_data(
        self,
        factors: list[Factor],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        載入因子資料和標籤

        Returns:
            DataFrame with columns: [factor1, factor2, ..., label]
            Index: MultiIndex (datetime, instrument)
        """
        # 強制重新初始化 qlib，確保使用最新導出的資料
        self._init_qlib(force=True)
        from qlib.data import D

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found in qlib data directory")

        # 構建因子表達式
        fields = [f.expression for f in factors]
        names = [f.name for f in factors]

        # 標籤：未來 1 日收益率
        label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
        all_fields = fields + [label_expr]
        all_names = names + ["label"]

        # 讀取資料
        df = D.features(
            instruments=instruments,
            fields=all_fields,
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return df

        # 重命名欄位
        df.columns = all_names

        return df

    def _prepare_train_valid_data(
        self,
        df: pd.DataFrame,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        準備訓練和驗證資料

        Returns:
            (X_train, X_valid, y_train, y_valid)
        """
        # 分離特徵和標籤
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols]
        y = df["label"]

        # 標籤截面標準化（仿 qlib CSZScoreNorm for label）
        # 讓模型學習相對排序而非絕對值
        y = self._zscore_by_date(y.to_frame()).squeeze()

        # 按日期分割
        train_mask = (df.index.get_level_values("datetime").date >= train_start) & \
                     (df.index.get_level_values("datetime").date <= train_end)
        valid_mask = (df.index.get_level_values("datetime").date >= valid_start) & \
                     (df.index.get_level_values("datetime").date <= valid_end)

        X_train = X[train_mask].dropna()
        X_valid = X[valid_mask].dropna()
        y_train = y[train_mask].dropna()
        y_valid = y[valid_mask].dropna()

        # 對齊索引
        common_train = X_train.index.intersection(y_train.index)
        common_valid = X_valid.index.intersection(y_valid.index)

        return (
            X_train.loc[common_train],
            X_valid.loc[common_valid],
            y_train.loc[common_train],
            y_valid.loc[common_valid],
        )

    def _process_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理無窮大值（仿 qlib ProcessInf）

        將 inf/-inf 替換為該欄位的均值
        """
        df = df.copy()
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
        return df

    def _zscore_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """每日截面標準化（仿 qlib CSZScoreNorm）"""
        return df.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        n_trials: int | None = None,
        timeout: int | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> dict:
        """
        使用 Optuna 優化 LightGBM 超參數

        Args:
            X_train, y_train: 訓練資料
            X_valid, y_valid: 驗證資料
            n_trials: 搜索次數（預設 OPTUNA_N_TRIALS）
            timeout: 超時秒數（預設 OPTUNA_TIMEOUT）
            on_progress: 進度回調

        Returns:
            最佳超參數字典
        """
        import optuna
        import lightgbm as lgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = n_trials or self.OPTUNA_N_TRIALS
        timeout = timeout or self.OPTUNA_TIMEOUT

        # 預處理資料（只做一次）
        X_train_processed = self._process_inf(X_train)
        X_valid_processed = self._process_inf(X_valid)
        X_train_norm = self._zscore_by_date(X_train_processed).fillna(0)
        X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)

        train_data = lgb.Dataset(X_train_norm.values, label=y_train.values)
        valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid.values, reference=train_data)

        # 計算數據特徵，用於動態設定搜索範圍
        n_samples = len(X_train)
        n_features = X_train.shape[1]

        # 動態計算搜索範圍
        max_leaves = min(256, max(32, int(n_samples ** 0.4)))
        max_min_data = max(100, n_samples // 100)

        best_ic = [0.0]  # 用 list 以便在閉包中修改
        trial_count = [0]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": 42,
                "feature_pre_filter": False,  # 允許動態調整 min_data_in_leaf
                # 搜索的超參數
                "num_leaves": trial.suggest_int("num_leaves", 16, max_leaves),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 50.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 50.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, max_min_data),
            }

            # 訓練模型
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ],
            )

            # 計算 IC
            predictions = model.predict(X_valid_norm.values)
            pred_df = pd.DataFrame({
                "pred": predictions,
                "label": y_valid.values,
            }, index=y_valid.index)

            daily_ic = pred_df.groupby(level="datetime").apply(
                lambda g: g["pred"].corr(g["label"]) if len(g) >= 2 else np.nan
            )
            mean_ic = daily_ic.mean()
            ic = float(mean_ic) if not np.isnan(mean_ic) else 0.0

            # 更新進度
            trial_count[0] += 1
            if ic > best_ic[0]:
                best_ic[0] = ic

            if on_progress:
                on_progress(
                    round(2.0 + (trial_count[0] / n_trials) * 8.0, 1),
                    f"Optuna trial {trial_count[0]}/{n_trials}: IC={ic:.4f} (best: {best_ic[0]:.4f})"
                )

            return ic

        # 執行優化
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        # 返回最佳參數
        best_params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
            **study.best_params,
        }

        return best_params

    def _train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        params: dict | None = None,
    ) -> Any:
        """
        訓練 LightGBM 模型

        Args:
            X_train, y_train: 訓練資料
            X_valid, y_valid: 驗證資料
            params: 超參數（若為 None，使用優化後的參數或預設值）

        Returns:
            訓練好的 LightGBM Booster
        """
        import lightgbm as lgb

        # 1. 處理無窮大值
        X_train = self._process_inf(X_train)
        X_valid = self._process_inf(X_valid)

        # 2. 每日截面標準化
        X_train_norm = self._zscore_by_date(X_train)
        X_valid_norm = self._zscore_by_date(X_valid)

        # 3. 填補 NaN（標準化後可能產生 NaN）
        X_train_norm = X_train_norm.fillna(0)
        X_valid_norm = X_valid_norm.fillna(0)

        # 建立 LightGBM Dataset
        train_data = lgb.Dataset(X_train_norm.values, label=y_train.values)
        valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid.values, reference=train_data)

        # 使用參數優先級：傳入參數 > 優化後參數 > 預設參數
        if params is None:
            params = self._optimized_params

        if params is None:
            # 預設參數（作為 fallback）
            params = {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "num_leaves": 64,
                "max_depth": 6,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 10.0,
                "lambda_l2": 10.0,
                "verbosity": -1,
                "seed": 42,
                "feature_pre_filter": False,
            }

        # 訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ],
        )

        return model

    def _calculate_prediction_ic(
        self,
        model: Any,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        """
        計算模型預測的 IC

        Returns:
            平均 IC 值
        """
        # 處理無窮大 + 標準化 + 填補 NaN
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed)
        X_valid_norm = X_valid_norm.fillna(0)

        # 預測
        predictions = model.predict(X_valid_norm.values)

        # 構建 DataFrame
        pred_df = pd.DataFrame({
            "pred": predictions,
            "label": y_valid.values,
        }, index=y_valid.index)

        # 計算每日截面 IC
        def calc_corr(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return np.nan
            return group["pred"].corr(group["label"])

        daily_ic = pred_df.groupby(level="datetime").apply(calc_corr)

        # 保存 IC 標準差
        self._last_ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else None

        mean_ic = daily_ic.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    def _calculate_daily_ic(
        self,
        model: Any,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> np.ndarray:
        """
        計算模型預測的每日 IC（用於統計檢驗）

        Returns:
            每日 IC 的 numpy 陣列
        """
        # 處理無窮大 + 標準化 + 填補 NaN
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed)
        X_valid_norm = X_valid_norm.fillna(0)

        # 預測
        predictions = model.predict(X_valid_norm.values)

        # 構建 DataFrame
        pred_df = pd.DataFrame({
            "pred": predictions,
            "label": y_valid.values,
        }, index=y_valid.index)

        # 計算每日截面 IC
        def calc_corr(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return np.nan
            return group["pred"].corr(group["label"])

        daily_ic = pred_df.groupby(level="datetime").apply(calc_corr)

        return daily_ic.dropna().values

    def _should_select_factor_with_test(
        self,
        ic_daily_new: np.ndarray,
        ic_daily_current: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[bool, dict]:
        """
        判斷是否應該選擇因子（複合動態閾值 + 雙重條件）

        基於 Harvey et al. (2016) 研究：296 個因子中 53% 是假發現，
        需要更嚴格的門檻來過濾冗餘因子。

        複合動態閾值設計：
        1. 相對百分比：不能下降超過當前 IC 的 10%
        2. σ 倍數：不能下降超過 0.2σ
        3. 取兩者中更嚴格者
        4. 安全範圍：-0.5% ~ -2%

        雙重條件（關鍵改進）：
        - 條件 1：CI 上界 > 閾值（非劣性檢驗）
        - 條件 2：mean_diff > 閾值（正向貢獻要求）

        Args:
            ic_daily_new: 新模型的每日 IC
            ic_daily_current: 當前模型的每日 IC
            alpha: 顯著性水準

        Returns:
            (should_select, test_result)
        """
        # 計算複合動態閾值
        current_ic = np.mean(ic_daily_current) if len(ic_daily_current) > 0 else 0.05
        ic_std = np.std(ic_daily_current) if len(ic_daily_current) > 1 else 0.02

        # 閾值 1：不能下降超過當前 IC 的 10%
        relative_threshold = -0.10 * abs(current_ic)

        # 閾值 2：不能下降超過 0.2σ（比原本的 0.5σ 更嚴格）
        std_threshold = -0.2 * ic_std

        # 取更嚴格者（數值更大的負數，即更接近 0）
        decline_threshold = max(relative_threshold, std_threshold)

        # 安全範圍：-0.5% ~ -2%
        decline_threshold = max(decline_threshold, -0.02)  # 下限：最多允許 2% 下降
        decline_threshold = min(decline_threshold, -0.005)  # 上限：至少允許 0.5% 下降

        # 確保長度一致
        min_len = min(len(ic_daily_new), len(ic_daily_current))
        if min_len < 5:
            # 資料太少，使用簡單比較
            mean_new = np.mean(ic_daily_new)
            mean_current = np.mean(ic_daily_current)
            mean_diff = mean_new - mean_current
            # 正向貢獻要求：mean_diff >= 0
            should_select = mean_diff >= 0
            return should_select, {
                'method': 'simple',
                'n_days': min_len,
                'mean_diff': float(mean_diff),
                'threshold': float(decline_threshold),
                'current_ic': float(current_ic),
                'ic_std': float(ic_std),
                'relative_threshold': float(relative_threshold),
                'std_threshold': float(std_threshold),
            }

        diff = ic_daily_new[:min_len] - ic_daily_current[:min_len]
        n = len(diff)
        mean_diff = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(n)

        if se == 0:
            # 雙重條件：需要 mean_diff >= 0
            should_select = mean_diff >= 0
            return should_select, {
                'method': 'constant_diff',
                'mean_diff': float(mean_diff),
                'threshold': float(decline_threshold),
            }

        # t-test
        t_stat = mean_diff / se

        # 計算 CI
        t_critical = stats.t.ppf(1 - alpha, n - 1)
        ci_upper = mean_diff + t_critical * se
        ci_lower = mean_diff - t_critical * se

        # 雙重條件（關鍵改進）：
        # 1. CI 上界 > 閾值：非劣性檢驗（統計上不顯著下降）
        # 2. mean_diff >= 0：正向貢獻要求（實際上沒有下降）
        # 注意：之前錯誤地用 mean_diff > threshold（負數），導致負貢獻的因子也被選中
        should_select = (ci_upper > decline_threshold) and (mean_diff >= 0)

        # 計算單尾 p-value
        t_stat_decline = (mean_diff - decline_threshold) / se
        p_value_decline = stats.t.cdf(t_stat_decline, n - 1)

        return should_select, {
            'method': 't_test_composite',
            'mean_diff': float(mean_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_stat': float(t_stat),
            'p_decline': float(p_value_decline),
            'threshold': float(decline_threshold),
            'current_ic': float(current_ic),
            'ic_std': float(ic_std),
            'relative_threshold': float(relative_threshold),
            'std_threshold': float(std_threshold),
            'n_days': n,
            'pass_ci': ci_upper > decline_threshold,
            'pass_mean': mean_diff >= 0,
        }

    def train(
        self,
        session: Session,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        week_id: str | None = None,
        factor_pool_hash: str | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """
        執行 LightGBM IC 增量選擇訓練（週訓練架構）

        流程：
        1. 訓練模型 A（train_start ~ train_end）
        2. 計算驗證期 IC（報告指標）
        3. 增量更新：A + 驗證期資料 → A'
        4. 保存 A'（部署用）

        Args:
            session: 資料庫 Session
            train_start: 訓練開始日期
            train_end: 訓練結束日期
            valid_start: 驗證開始日期
            valid_end: 驗證結束日期
            week_id: 週 ID（如 "2026W05"）
            factor_pool_hash: 因子池 hash
            on_progress: 進度回調 (progress: 0-100, message: str)

        Returns:
            TrainingResult
        """
        from src.shared.constants import EMBARGO_DAYS

        factor_repo = FactorRepository(session)
        training_repo = TrainingRepository(session)

        # 取得啟用的因子
        enabled_factors = factor_repo.get_all(enabled=True)
        if not enabled_factors:
            raise ValueError("No enabled factors found")

        candidate_ids = [f.id for f in enabled_factors]

        # 生成臨時模型名稱
        temp_name = f"{week_id or valid_end.strftime('%Y%m')}-pending"

        # 創建訓練記錄（含週訓練相關欄位）
        run = training_repo.create_run(
            name=temp_name,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            week_id=week_id,
            factor_pool_hash=factor_pool_hash,
            embargo_days=EMBARGO_DAYS,
        )
        run.candidate_factor_ids = json.dumps(candidate_ids)
        run.status = "running"
        session.commit()

        if on_progress:
            on_progress(0.0, "Initializing training...")

        try:
            # 預載入所有因子資料
            if on_progress:
                on_progress(2.0, "Loading factor data...")

            all_data = self._load_data(
                factors=enabled_factors,
                start_date=train_start,
                end_date=valid_end,
            )

            if all_data.empty:
                raise ValueError("No data available for the specified date range")

            # === 超參數載入（從培養文件）===
            cultivated_params = self.load_cultivated_hyperparameters()
            if cultivated_params and on_progress:
                on_progress(10.0, "Using pre-cultivated hyperparameters")

            if cultivated_params:
                self._optimized_params = cultivated_params
            else:
                # 無培養超參數，執行 Optuna 優化
                if on_progress:
                    on_progress(2.5, "No cultivated params, running Optuna...")

                # 準備優化用的資料（使用所有因子）
                factor_names = [f.name for f in enabled_factors]
                X_all = all_data[factor_names]
                y_all = all_data["label"]

                X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = self._prepare_train_valid_data(
                    pd.concat([X_all, y_all], axis=1),
                    train_start, train_end, valid_start, valid_end,
                )

                if not X_train_opt.empty and not X_valid_opt.empty:
                    self._optimized_params = self._optimize_hyperparameters(
                        X_train_opt, y_train_opt,
                        X_valid_opt, y_valid_opt,
                        on_progress=on_progress,
                    )
                    if on_progress:
                        on_progress(10.0, f"Optuna done: best params found")
                else:
                    self._optimized_params = None
                    if on_progress:
                        on_progress(10.0, "Skipped Optuna (insufficient data)")

            # 執行 IC 增量選擇（使用優化後的參數，並根據因子數動態調整）
            selected_factors, all_results, best_model, selection_stats = self._incremental_ic_selection(
                factors=enabled_factors,
                all_data=all_data,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                on_progress=on_progress,
                base_factor_count=len(enabled_factors),  # 培養時使用的因子數
            )

            # 計算最終模型 IC（使用最佳模型的 IC）
            # 注意：必須使用 selected_factors 的原始順序，因為 LightGBM 按位置識別特徵
            if best_model is not None and selected_factors:
                factor_names = [f.name for f in selected_factors]
                X_valid = all_data[factor_names]
                y_valid = all_data["label"]

                # 分割驗證資料
                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                             (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid = X_valid[valid_mask].dropna()
                y_valid = y_valid[valid_mask].dropna()
                common_idx = X_valid.index.intersection(y_valid.index)
                X_valid = X_valid.loc[common_idx]
                y_valid = y_valid.loc[common_idx]

                model_ic = self._calculate_prediction_ic(best_model, X_valid, y_valid)
            else:
                model_ic = 0.0

            # 計算 ICIR
            icir = self._calculate_icir(model_ic, len(selected_factors))

            # 保存因子結果
            for result in all_results:
                training_repo.add_factor_result(
                    run_id=run.id,
                    factor_id=result.factor_id,
                    ic_value=result.ic_value,
                    selected=result.selected,
                )

            # 生成最終模型名稱：{week_id}-{factor_pool_hash}
            # 若無 week_id，使用舊格式 YYYYMM-hash
            if week_id and factor_pool_hash:
                model_name = f"{week_id}-{factor_pool_hash}"
            else:
                hash_input = f"{run.id}-{valid_end.isoformat()}-{len(selected_factors)}"
                short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
                model_name = f"{valid_end.strftime('%Y%m')}-{short_hash}"

            # === 增量更新（驗證後重訓）===
            # 根據設計文檔：訓練完成後，用驗證期資料做增量更新
            # 保存的是 A'（增量更新後的模型），但報告的是 A 的 IC
            import lightgbm as lgb

            incremented_model = best_model
            if best_model is not None and selected_factors:
                if on_progress:
                    on_progress(96.0, "Incremental update with validation data...")

                factor_names = [f.name for f in selected_factors]
                X_valid_incr = all_data[factor_names]
                y_valid_incr = all_data["label"]

                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                             (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid_incr = X_valid_incr[valid_mask].dropna()
                y_valid_incr = y_valid_incr[valid_mask].dropna()
                common_idx = X_valid_incr.index.intersection(y_valid_incr.index)
                X_valid_incr = X_valid_incr.loc[common_idx]
                y_valid_incr = y_valid_incr.loc[common_idx]

                if not X_valid_incr.empty:
                    # 處理和標準化
                    X_valid_processed = self._process_inf(X_valid_incr)
                    X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
                    y_valid_zscore = self._zscore_by_date(y_valid_incr.to_frame()).squeeze()

                    # 增量更新：使用 init_model
                    valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid_zscore.values)

                    # 使用相同參數，但從 best_model 開始
                    incr_params = self._optimized_params or {
                        "objective": "regression",
                        "metric": "mse",
                        "boosting_type": "gbdt",
                        "verbosity": -1,
                        "seed": 42,
                    }

                    try:
                        incremented_model = lgb.train(
                            incr_params,
                            valid_data,
                            num_boost_round=50,  # 少量更新
                            init_model=best_model,
                            keep_training_booster=True,
                        )
                        if on_progress:
                            on_progress(98.0, "Incremental update completed")
                    except Exception as e:
                        # 增量更新失敗，使用原模型
                        if on_progress:
                            on_progress(98.0, f"Incremental update failed: {e}, using original model")
                        incremented_model = best_model

            # 完成訓練
            run.name = model_name
            run.selected_factor_ids = json.dumps([f.id for f in selected_factors])

            # 記錄因子選擇策略（用於追踪改動效果）
            selection_config = {
                "relative_pct": 0.10,
                "std_multiplier": 0.2,
                "threshold_min": -0.02,
                "threshold_max": -0.005,
                "require_positive_contribution": True,
                "require_positive_single_ic": True,
                "incremental_update": True,  # 標記有做增量更新
            }
            run.selection_method = "composite_threshold_v3"
            run.selection_config = json.dumps(selection_config)
            run.selection_stats = json.dumps(selection_stats)

            training_repo.complete_run(
                run_id=run.id,
                model_ic=model_ic,  # 報告的是模型 A 的 IC（驗證期 IC）
                icir=icir,
                factor_count=len(selected_factors),
            )

            # 保存模型檔案（保存增量更新後的 A'）
            config = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
                "week_id": week_id,
                "factor_pool_hash": factor_pool_hash,
                "model_ic": model_ic,  # 報告的 IC（模型 A）
                "icir": icir,
                "incremental_updated": incremented_model is not best_model,
            }
            if self._optimized_params:
                tuned_params = {k: v for k, v in self._optimized_params.items()
                               if k not in ("objective", "metric", "boosting_type", "verbosity", "seed")}
                config["hyperparameters"] = tuned_params

            self._save_model_files(
                model_name=model_name,
                selected_factors=selected_factors,
                config=config,
                model=incremented_model,  # 保存增量更新後的模型
            )

            if on_progress:
                on_progress(100.0, "Training completed")

            return TrainingResult(
                run_id=run.id,
                model_name=model_name,
                model_ic=model_ic,
                icir=icir,
                selected_factor_ids=[f.id for f in selected_factors],
                all_results=all_results,
            )

        except Exception as e:
            # 標記訓練失敗
            run.status = "failed"
            run.completed_at = datetime.now(TZ_TAIPEI)
            session.commit()
            raise e

    def _calculate_single_factor_ic(
        self,
        factor: Factor,
        all_data: pd.DataFrame,
        train_start: date,
        train_end: date,
    ) -> float:
        """
        計算單因子 IC（用於排序）

        注意：只使用訓練期資料，避免資料洩漏
        """
        try:
            # 只使用訓練期資料（不可偷看驗證期）
            train_mask = (all_data.index.get_level_values("datetime").date >= train_start) & \
                         (all_data.index.get_level_values("datetime").date <= train_end)

            factor_data = all_data[[factor.name, "label"]][train_mask].dropna()
            if len(factor_data) < 100:
                return 0.0

            # 計算每日截面 IC（因子值與標籤的相關性）
            def calc_corr(group: pd.DataFrame) -> float:
                if len(group) < 5:
                    return np.nan
                return group[factor.name].corr(group["label"])

            daily_ic = factor_data.groupby(level="datetime").apply(calc_corr)
            mean_ic = daily_ic.mean()
            return float(mean_ic) if not np.isnan(mean_ic) else 0.0
        except Exception:
            return 0.0

    def _incremental_ic_selection(
        self,
        factors: list[Factor],
        all_data: pd.DataFrame,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
        base_factor_count: int | None = None,
    ) -> tuple[list[Factor], list[FactorEvalResult], Any, dict]:
        """
        LightGBM IC 增量選擇法（含因子數量自適應參數）

        1. 先計算每個因子的單獨 IC
        2. 按 IC 絕對值降序排列（高預測力因子優先）
        3. 依序測試加入，若模型 IC 提升則納入
        4. 超參數會根據當前因子數量動態調整

        Args:
            base_factor_count: 培養超參數時使用的因子數（用於縮放）

        Returns:
            (selected_factors, all_results, best_model, selection_stats)
            注意：selected_factors 保持選擇順序，這對模型預測至關重要
        """
        # 確定基準因子數（培養時的因子數）
        if base_factor_count is None:
            base_factor_count = len(factors)
        if on_progress:
            on_progress(11.0, "Calculating single-factor ICs...")

        # 計算每個因子的單獨 IC（只用訓練期，避免資料洩漏）
        factor_ics: list[tuple[Factor, float]] = []
        for factor in factors:
            ic = self._calculate_single_factor_ic(factor, all_data, train_start, train_end)
            factor_ics.append((factor, ic))

        # 按 IC 絕對值降序排列（高預測力因子優先測試）
        sorted_factors = sorted(factor_ics, key=lambda x: abs(x[1]), reverse=True)

        if on_progress:
            on_progress(13.0, f"Sorted {len(sorted_factors)} factors by IC")

        selected_factors: list[Factor] = []
        all_results: list[FactorEvalResult] = []
        current_ic = 0.0
        best_model = None

        # 收集選擇統計
        thresholds_used: list[float] = []
        mean_diffs: list[float] = []

        total = len(sorted_factors)
        for i, (factor, single_ic) in enumerate(sorted_factors):
            # 進度：13% 初始化 + 82% 因子挑選（10% Optuna + 3% 單因子 IC）
            if on_progress:
                progress = round(13.0 + (i / total) * 82.0, 1)
                on_progress(progress, f"Evaluating ({i+1}/{total}): {factor.name} (single IC: {single_ic:.4f})")

            # 測試加入此因子
            test_factors = selected_factors + [factor]
            test_factor_names = [f.name for f in test_factors]

            # 準備資料
            X = all_data[test_factor_names]
            y = all_data["label"]

            X_train, X_valid, y_train, y_valid = self._prepare_train_valid_data(
                pd.concat([X, y], axis=1),
                train_start, train_end, valid_start, valid_end,
            )

            if X_train.empty or X_valid.empty:
                # 資料不足，跳過此因子
                all_results.append(
                    FactorEvalResult(
                        factor_id=factor.id,
                        factor_name=factor.name,
                        ic_value=0.0,
                        selected=False,
                    )
                )
                continue

            # 根據當前因子數量調整超參數
            current_factor_count = len(test_factors)
            base_params = self._optimized_params or get_conservative_default_params(current_factor_count)
            adapted_params = scale_params_for_factor_count(
                base_params,
                base_factor_count=base_factor_count,
                actual_factor_count=current_factor_count,
            )

            # 訓練 LightGBM（使用調整後的參數）
            try:
                model = self._train_lgbm(X_train, y_train, X_valid, y_valid, params=adapted_params)
                new_ic = self._calculate_prediction_ic(model, X_valid, y_valid)
                new_daily_ic = self._calculate_daily_ic(model, X_valid, y_valid)
            except Exception as e:
                # 訓練失敗，跳過此因子
                all_results.append(
                    FactorEvalResult(
                        factor_id=factor.id,
                        factor_name=factor.name,
                        ic_value=0.0,
                        selected=False,
                    )
                )
                continue

            # 使用統計檢驗判斷是否選擇（解決假性低 IC 問題）
            if best_model is None:
                # 第一個因子：只要 IC > 0 就入選
                should_select = new_ic > 0
                test_result = {'method': 'first_factor'}
            else:
                # 計算當前最佳模型的每日 IC
                # 注意：需要使用相同的驗證資料（已選因子）
                prev_factor_names = [f.name for f in selected_factors]
                X_valid_prev = all_data[prev_factor_names]
                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                             (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid_prev = X_valid_prev[valid_mask].dropna()
                y_valid_prev = all_data["label"][valid_mask].dropna()
                common_idx_prev = X_valid_prev.index.intersection(y_valid_prev.index)
                X_valid_prev = X_valid_prev.loc[common_idx_prev]
                y_valid_prev = y_valid_prev.loc[common_idx_prev]

                prev_daily_ic = self._calculate_daily_ic(best_model, X_valid_prev, y_valid_prev)

                # 統計檢驗
                should_select, test_result = self._should_select_factor_with_test(
                    new_daily_ic, prev_daily_ic, alpha=0.10
                )

                # 收集統計
                if 'threshold' in test_result:
                    thresholds_used.append(test_result['threshold'])
                if 'mean_diff' in test_result:
                    mean_diffs.append(test_result['mean_diff'])

            # 額外條件：單因子 IC 必須為正
            # 文獻建議：確保每個因子本身有預測能力，避免噪音因子
            if should_select and single_ic <= 0:
                should_select = False

            if should_select:
                selected_factors.append(factor)
                current_ic = new_ic
                best_model = model
                selected = True
            else:
                selected = False

            # 記錄結果（使用單因子 IC，更能反映因子品質）
            all_results.append(
                FactorEvalResult(
                    factor_id=factor.id,
                    factor_name=factor.name,
                    ic_value=single_ic,
                    selected=selected,
                )
            )

            # 評估完成後更新進度
            if on_progress:
                progress = round(13.0 + ((i + 1) / total) * 82.0, 1)
                status = "selected" if selected else "skipped"
                p_info = f", p={test_result.get('p_value', 'N/A'):.3f}" if 'p_value' in test_result else ""
                on_progress(progress, f"Factor {factor.name}: {status} (IC: {new_ic:.4f}{p_info})")

        # 計算選擇統計
        n_selected = len(selected_factors)
        n_rejected = total - n_selected
        selection_stats = {
            "candidates_evaluated": total,
            "factors_selected": n_selected,
            "factors_rejected": n_rejected,
            "selection_rate": round(n_selected / total, 4) if total > 0 else 0,
            "avg_threshold": round(np.mean(thresholds_used), 6) if thresholds_used else None,
            "avg_mean_diff": round(np.mean(mean_diffs), 6) if mean_diffs else None,
        }

        # 返回 selected_factors 列表（保持順序），不是 set
        # 順序對 LightGBM 預測至關重要，因為模型用 .values 訓練，只認位置不認名稱
        return selected_factors, all_results, best_model, selection_stats

    def _calculate_icir(self, ic: float, factor_count: int) -> float | None:
        """計算 IC Information Ratio (ICIR = IC / std(IC))"""
        if factor_count == 0 or ic == 0:
            return None

        ic_std = self._last_ic_std
        if ic_std is None or ic_std == 0:
            return None

        return ic / ic_std

    def _save_model_files(
        self,
        model_name: str,
        selected_factors: list[Factor],
        config: dict,
        model: Any = None,
    ) -> None:
        """保存模型檔案（含 LightGBM .pkl）"""
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = model_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存因子
        factors_data = [
            {
                "id": f.id,
                "name": f.name,
                "expression": f.expression,
            }
            for f in selected_factors
        ]
        factors_path = model_dir / "factors.json"
        with open(factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f, indent=2, ensure_ascii=False)

        # 保存 LightGBM 模型
        if model is not None:
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    # === 超參數培養 ===

    def _generate_walk_forward_periods(
        self,
        data_start: date,
        data_end: date,
        n_periods: int = 5,
        train_days: int | None = None,
        valid_days: int | None = None,
    ) -> list[tuple[date, date, date, date]]:
        """
        生成 Walk Forward 窗口

        Args:
            data_start: 資料起始日
            data_end: 資料結束日
            n_periods: 期望窗口數量
            train_days: 訓練天數（預設使用 TRAIN_DAYS）
            valid_days: 驗證天數（預設使用 VALID_DAYS）

        Returns:
            list of (train_start, train_end, valid_start, valid_end)
        """
        train_days = train_days or TRAIN_DAYS
        valid_days = valid_days or VALID_DAYS

        total_days = (data_end - data_start).days
        period_length = train_days + valid_days  # 訓練期 + 驗證期

        # 確保有足夠的資料
        if total_days < period_length:
            raise ValueError(
                f"Insufficient data: need {period_length} days, have {total_days}"
            )

        # 計算可用的窗口數量
        available_periods = (total_days - train_days) // valid_days
        actual_periods = min(n_periods, max(1, available_periods))

        periods = []
        # 從最新資料開始向前推算
        for i in range(actual_periods):
            # 驗證期結束：從 data_end 向前推 i 個驗證期
            valid_end = data_end - timedelta(days=i * valid_days)
            valid_start = valid_end - timedelta(days=valid_days - 1)
            train_end = valid_start - timedelta(days=1)
            train_start = train_end - timedelta(days=train_days - 1)

            # 確保訓練起始日不早於資料起始日
            if train_start < data_start:
                break

            periods.append((train_start, train_end, valid_start, valid_end))

        # 反轉使最早的窗口在前面（方便閱讀）
        return list(reversed(periods))

    def _aggregate_median(self, all_params: list[dict]) -> dict:
        """
        聚合多個窗口的參數，取中位數

        Args:
            all_params: 各窗口最佳參數列表

        Returns:
            中位數參數
        """
        if not all_params:
            return {}

        # 需要取中位數的參數（數值型）
        numeric_params = [
            "num_leaves", "max_depth", "learning_rate", "feature_fraction",
            "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
            "min_data_in_leaf"
        ]

        result = {}
        for param in numeric_params:
            values = [p.get(param) for p in all_params if param in p]
            if values:
                if isinstance(values[0], int):
                    result[param] = int(statistics.median(values))
                else:
                    result[param] = float(statistics.median(values))

        # 固定參數
        result.update({
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
        })

        return result

    def _calculate_stability(self, all_params: list[dict]) -> dict[str, float]:
        """
        計算參數穩定性（變異係數 CV = std / mean）

        Returns:
            各參數的 CV 值（CV < 0.3 表示穩定）
        """
        if len(all_params) < 2:
            return {}

        numeric_params = [
            "num_leaves", "max_depth", "learning_rate", "feature_fraction",
            "bagging_fraction", "lambda_l1", "lambda_l2", "min_data_in_leaf"
        ]

        stability = {}
        for param in numeric_params:
            values = [p.get(param) for p in all_params if param in p]
            if len(values) >= 2:
                mean = statistics.mean(values)
                if mean != 0:
                    std = statistics.stdev(values)
                    stability[param] = round(std / abs(mean), 3)
                else:
                    stability[param] = 0.0

        return stability

    def cultivate_hyperparameters(
        self,
        factors: list[Factor],
        n_periods: int = 5,
        n_trials_per_period: int = 20,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> CultivationResult:
        """
        執行多窗口超參數培養

        基於 Walk Forward Optimization + Median Aggregation 方法：
        1. 生成多個歷史窗口
        2. 每個窗口獨立運行 Optuna
        3. 取各窗口最佳參數的中位數
        4. 計算穩定性指標 (CV)

        Args:
            factors: 因子列表
            n_periods: 窗口數量（建議 5）
            n_trials_per_period: 每窗口 Optuna 試驗次數（建議 20）
            on_progress: 進度回調

        Returns:
            CultivationResult
        """
        self._init_qlib()

        if on_progress:
            on_progress(0.0, "Initializing hyperparameter cultivation...")

        # 取得資料日期範圍
        data_start, data_end = self.get_data_date_range()
        if not data_start or not data_end:
            raise ValueError("No qlib data found")

        # 動態計算 train_days 和 valid_days 以符合請求的 n_periods
        # 目標：訓練期佔 80%，驗證期佔 20%
        total_days = (data_end - data_start).days

        # 每個 period 需要的總天數 = total_days / n_periods
        # 但 periods 會重疊（訓練期重疊），所以計算方式是：
        # total_days = train_days + valid_days * n_periods
        # => valid_days = (total_days - train_days) / n_periods
        # 設 train_days = 504 (2年)，計算 valid_days
        MIN_TRAIN_DAYS = 378  # 最少 1.5 年
        MAX_TRAIN_DAYS = 630  # 最多 2.5 年
        MIN_VALID_DAYS = 42   # 最少 2 個月
        MAX_VALID_DAYS = 126  # 最多 6 個月

        # 嘗試計算可行的 train_days 和 valid_days
        train_days = 504  # 預設 2 年
        valid_days = (total_days - train_days) // n_periods

        # 如果 valid_days 太小，減少 train_days
        if valid_days < MIN_VALID_DAYS:
            valid_days = MIN_VALID_DAYS
            train_days = total_days - valid_days * n_periods
            if train_days < MIN_TRAIN_DAYS:
                raise ValueError(
                    f"Insufficient data for {n_periods} periods. "
                    f"Need at least {MIN_TRAIN_DAYS + MIN_VALID_DAYS * n_periods} days, have {total_days}."
                )

        # 如果 valid_days 太大，限制它
        if valid_days > MAX_VALID_DAYS:
            valid_days = MAX_VALID_DAYS

        # 確保 train_days 在範圍內
        train_days = max(MIN_TRAIN_DAYS, min(MAX_TRAIN_DAYS, train_days))

        if on_progress:
            on_progress(1.0, f"Using {train_days} train days, {valid_days} valid days per period")

        periods = self._generate_walk_forward_periods(
            data_start, data_end, n_periods,
            train_days=train_days,
            valid_days=valid_days,
        )

        if len(periods) < n_periods:
            raise ValueError(
                f"Could only generate {len(periods)} periods (requested {n_periods}). "
                f"Need more data or reduce n_periods."
            )

        if on_progress:
            on_progress(2.0, f"Generated {len(periods)} walk-forward periods")

        # 載入所有資料（一次載入，避免重複讀取）
        if on_progress:
            on_progress(3.0, "Loading factor data...")

        all_data = self._load_data(
            factors=factors,
            start_date=data_start,
            end_date=data_end,
        )

        if all_data.empty:
            raise ValueError("No data available for cultivation")

        # 各窗口優化
        all_params = []
        period_results = []

        for i, (train_start, train_end, valid_start, valid_end) in enumerate(periods):
            period_name = f"{train_start.strftime('%Y-%m')} ~ {valid_end.strftime('%Y-%m')}"

            if on_progress:
                base_progress = 5.0 + (i / len(periods)) * 85.0
                on_progress(base_progress, f"Optimizing period {i+1}/{len(periods)}: {period_name}")

            # 準備該窗口的資料
            factor_names = [f.name for f in factors]
            X = all_data[factor_names]
            y = all_data["label"]

            X_train, X_valid, y_train, y_valid = self._prepare_train_valid_data(
                pd.concat([X, y], axis=1),
                train_start, train_end, valid_start, valid_end,
            )

            if X_train.empty or X_valid.empty:
                if on_progress:
                    on_progress(base_progress + 3, f"Skipped period {i+1} (insufficient data)")
                continue

            # Optuna 優化
            # 每個 period 佔用 85% / n_periods 的進度
            period_progress_range = 85.0 / len(periods)

            def period_progress(p: float, msg: str) -> None:
                if on_progress:
                    # p 是 2~10 的進度（Optuna 部分）
                    # 轉換為 0~1 範圍，再映射到這個 period 的進度區間
                    optuna_progress = (p - 2.0) / 8.0  # 0~1
                    inner_progress = base_progress + optuna_progress * period_progress_range
                    on_progress(inner_progress, f"[Period {i+1}/{len(periods)}] {msg}")

            # timeout 根據 trials 數量調整（每 trial 約 15 秒）
            timeout_seconds = max(300, n_trials_per_period * 15)

            best_params = self._optimize_hyperparameters(
                X_train, y_train, X_valid, y_valid,
                n_trials=n_trials_per_period,
                timeout=timeout_seconds,
                on_progress=period_progress,
            )

            # 計算最佳 IC
            import lightgbm as lgb
            X_train_norm = self._zscore_by_date(self._process_inf(X_train)).fillna(0)
            X_valid_norm = self._zscore_by_date(self._process_inf(X_valid)).fillna(0)
            train_data = lgb.Dataset(X_train_norm.values, label=y_train.values)
            valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid.values, reference=train_data)

            model = lgb.train(
                best_params, train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            predictions = model.predict(X_valid_norm.values)
            pred_df = pd.DataFrame({
                "pred": predictions, "label": y_valid.values
            }, index=y_valid.index)
            daily_ic = pred_df.groupby(level="datetime").apply(
                lambda g: g["pred"].corr(g["label"]) if len(g) >= 2 else np.nan
            )
            best_ic = float(daily_ic.mean())

            # 只保存調參的部分
            tuned_params = {k: v for k, v in best_params.items()
                           if k not in ("objective", "metric", "boosting_type", "verbosity", "seed", "feature_pre_filter")}
            all_params.append(tuned_params)

            period_results.append(PeriodResult(
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                best_ic=best_ic,
                params=tuned_params,
            ))

            if on_progress:
                on_progress(
                    5.0 + ((i + 1) / len(periods)) * 85.0,
                    f"Period {i+1} done: IC={best_ic:.4f}"
                )

        if not all_params:
            raise ValueError("No successful optimization in any period")

        # 聚合中位數
        if on_progress:
            on_progress(92.0, "Aggregating parameters...")

        final_params = self._aggregate_median(all_params)
        stability = self._calculate_stability(all_params)

        # 保存到檔案
        if on_progress:
            on_progress(95.0, "Saving hyperparameters...")

        result = CultivationResult(
            cultivated_at=datetime.now(TZ_TAIPEI).isoformat(),
            n_periods=len(period_results),
            params=final_params,
            stability=stability,
            periods=period_results,
        )

        # 注意：不再保存到檔案，由 API 層存入資料庫

        if on_progress:
            on_progress(100.0, f"Cultivation complete: {len(period_results)} periods")

        return result

    def _save_cultivated_hyperparameters(self, result: CultivationResult) -> None:
        """保存培養結果到檔案"""
        HYPERPARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "cultivated_at": result.cultivated_at,
            "n_periods": result.n_periods,
            "params": result.params,
            "stability": result.stability,
            "periods": [
                {
                    "train_start": p.train_start.isoformat(),
                    "train_end": p.train_end.isoformat(),
                    "valid_start": p.valid_start.isoformat(),
                    "valid_end": p.valid_end.isoformat(),
                    "best_ic": p.best_ic,
                    "params": p.params,
                }
                for p in result.periods
            ],
        }

        with open(HYPERPARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_cultivated_hyperparameters() -> dict | None:
        """
        載入已培養的超參數

        Returns:
            超參數字典，若檔案不存在則返回 None
        """
        if not HYPERPARAMS_FILE.exists():
            return None

        with open(HYPERPARAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("params")

    @staticmethod
    def get_cultivation_info() -> dict | None:
        """
        取得培養資訊（含穩定性指標）

        Returns:
            完整培養資訊，若檔案不存在則返回 None
        """
        if not HYPERPARAMS_FILE.exists():
            return None

        with open(HYPERPARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def run_training(
    session: Session,
    qlib_data_dir: Path | str,
    train_end: date | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> TrainingResult:
    """
    執行訓練的便利函數

    Args:
        session: 資料庫 Session
        qlib_data_dir: qlib 資料目錄
        train_end: 訓練結束日期（預設：今日 - 驗證期天數）
        on_progress: 進度回調
    """
    from datetime import timedelta

    today = date.today()

    if train_end is None:
        train_end = today - timedelta(days=VALID_DAYS)

    train_start = train_end - timedelta(days=TRAIN_DAYS)
    valid_start = train_end + timedelta(days=1)
    valid_end = today

    trainer = ModelTrainer(qlib_data_dir)
    return trainer.train(
        session=session,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        on_progress=on_progress,
    )
