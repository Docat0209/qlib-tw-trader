"""
模型訓練服務 - LightGBM IC 增量選擇法
"""

import json
import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor, TrainingRun
from src.repositories.training import TrainingRepository
from src.shared.constants import TRAIN_DAYS, VALID_DAYS

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# 模型檔案目錄
MODELS_DIR = Path("data/models")


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


class ModelTrainer:
    """模型訓練器 - LightGBM IC 增量選擇法"""

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False
        self._last_ic_std: float | None = None  # 用於 ICIR 計算
        self._data_cache: dict[str, pd.DataFrame] = {}  # 資料快取

    def _init_qlib(self) -> None:
        """初始化 qlib"""
        if self._qlib_initialized:
            return

        try:
            import qlib
            from qlib.config import REG_CN

            qlib.init(
                provider_uri=str(self.data_dir),
                region=REG_CN,
            )
            self._qlib_initialized = True
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
        self._init_qlib()
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

    def _train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Any:
        """
        訓練 LightGBM 模型

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

        # 訓練參數
        # 注意：qlib 官方參數針對大數據集（7年、300檔、158因子）
        # 我們的數據集較小（3年、100檔、30因子），需要調整
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            # 樹結構（適度複雜）
            "num_leaves": 64,
            "max_depth": 6,
            "learning_rate": 0.05,
            # 抽樣
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            # L1/L2 正則化（適度，避免過擬合）
            "lambda_l1": 10.0,
            "lambda_l2": 10.0,
            # 其他
            "verbose": -1,
            "seed": 42,
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

    def train(
        self,
        session: Session,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """
        執行 LightGBM IC 增量選擇訓練

        Args:
            session: 資料庫 Session
            train_start: 訓練開始日期
            train_end: 訓練結束日期
            valid_start: 驗證開始日期
            valid_end: 驗證結束日期
            on_progress: 進度回調 (progress: 0-100, message: str)

        Returns:
            TrainingResult
        """
        factor_repo = FactorRepository(session)
        training_repo = TrainingRepository(session)

        # 取得啟用的因子
        enabled_factors = factor_repo.get_all(enabled=True)
        if not enabled_factors:
            raise ValueError("No enabled factors found")

        candidate_ids = [f.id for f in enabled_factors]

        # 生成模型名稱
        model_name = f"{train_start.strftime('%Y-%m')}~{train_end.strftime('%Y-%m')}"

        # 創建訓練記錄
        run = training_repo.create_run(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        run.name = model_name
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

            # 執行 IC 增量選擇
            selected_factors, all_results, best_model = self._incremental_ic_selection(
                factors=enabled_factors,
                all_data=all_data,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                on_progress=on_progress,
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

            # 完成訓練
            run.selected_factor_ids = json.dumps([f.id for f in selected_factors])
            training_repo.complete_run(
                run_id=run.id,
                model_ic=model_ic,
                icir=icir,
                factor_count=len(selected_factors),
            )

            # 保存模型檔案（必須使用原始 selected_factors 保持順序）
            self._save_model_files(
                model_name=model_name,
                selected_factors=selected_factors,
                config={
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "valid_start": valid_start.isoformat(),
                    "valid_end": valid_end.isoformat(),
                    "model_ic": model_ic,
                    "icir": icir,
                },
                model=best_model,
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
    ) -> tuple[list[Factor], list[FactorEvalResult], Any]:
        """
        LightGBM IC 增量選擇法

        1. 先計算每個因子的單獨 IC
        2. 按 IC 絕對值降序排列（高預測力因子優先）
        3. 依序測試加入，若模型 IC 提升則納入

        Returns:
            (selected_factors, all_results, best_model)
            注意：selected_factors 保持選擇順序，這對模型預測至關重要
        """
        if on_progress:
            on_progress(3.0, "Calculating single-factor ICs...")

        # 計算每個因子的單獨 IC（只用訓練期，避免資料洩漏）
        factor_ics: list[tuple[Factor, float]] = []
        for factor in factors:
            ic = self._calculate_single_factor_ic(factor, all_data, train_start, train_end)
            factor_ics.append((factor, ic))

        # 按 IC 絕對值降序排列（高預測力因子優先測試）
        sorted_factors = sorted(factor_ics, key=lambda x: abs(x[1]), reverse=True)

        if on_progress:
            on_progress(5.0, f"Sorted {len(sorted_factors)} factors by IC")

        selected_factors: list[Factor] = []
        all_results: list[FactorEvalResult] = []
        current_ic = 0.0
        best_model = None

        total = len(sorted_factors)
        for i, (factor, single_ic) in enumerate(sorted_factors):
            # 進度：5% 初始化 + 90% 因子挑選
            if on_progress:
                progress = round(5.0 + (i / total) * 90.0, 1)
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

            # 訓練 LightGBM
            try:
                model = self._train_lgbm(X_train, y_train, X_valid, y_valid)
                new_ic = self._calculate_prediction_ic(model, X_valid, y_valid)
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

            # 嚴格大於才入選
            if new_ic > current_ic:
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
                progress = round(5.0 + ((i + 1) / total) * 90.0, 1)
                status = "selected" if selected else "skipped"
                on_progress(progress, f"Factor {factor.name}: {status} (IC: {new_ic:.4f})")

        # 返回 selected_factors 列表（保持順序），不是 set
        # 順序對 LightGBM 預測至關重要，因為模型用 .values 訓練，只認位置不認名稱
        return selected_factors, all_results, best_model

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
