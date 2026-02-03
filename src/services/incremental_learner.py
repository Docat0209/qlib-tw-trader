"""
增量學習服務 - 使用 LightGBM init_model 進行模型增量更新
"""

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from sqlalchemy.orm import Session

from src.shared.constants import LABEL_EXPR, QLIB_DATA_DIR


class IncrementalLearner:
    """
    增量學習服務

    使用 LightGBM 的 init_model 參數進行模型增量更新。
    核心原理：因子結構變化慢（週/月），因子權重變化快（日），
    透過增量學習快速調整權重以適應最新市場狀態。
    """

    def __init__(self, session: Session):
        self._session = session
        self._qlib_initialized = False
        self._qlib_data_dir = Path(QLIB_DATA_DIR)

    def _init_qlib(self):
        """初始化 Qlib"""
        if self._qlib_initialized:
            return

        qlib.init(
            provider_uri=str(self._qlib_data_dir),
            region="tw",
        )
        self._qlib_initialized = True

    def _get_instruments(self) -> list[str]:
        """取得股票清單"""
        instruments_file = self._qlib_data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        return []

    def _process_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理無窮大值"""
        df = df.copy()
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
        return df

    def _zscore_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """每日截面標準化"""
        return df.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _load_data(
        self,
        factors: list[dict],
        start_date: date,
        end_date: date,
    ) -> tuple[pd.DataFrame, pd.Series] | None:
        """
        載入指定期間的特徵和標籤

        Args:
            factors: 因子列表 [{"name": "...", "expression": "..."}]
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            (X, y) 或 None 如果無資料
        """
        self._init_qlib()

        instruments = self._get_instruments()
        if not instruments:
            return None

        fields = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        # 載入特徵
        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return None

        df.columns = names

        # 載入標籤
        label_df = D.features(
            instruments=instruments,
            fields=[LABEL_EXPR],
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d"),
        )

        if label_df.empty:
            return None

        label_df.columns = ["label"]

        # 合併並對齊
        merged = df.join(label_df, how="inner")
        merged = merged.dropna(subset=["label"])

        if len(merged) < 100:  # 最少需要 100 筆資料
            return None

        X = merged.drop(columns=["label"])
        y = merged["label"]

        # 處理數據
        X = self._process_inf(X)
        X = self._zscore_by_date(X)
        X = X.fillna(0)

        return X, y

    def update(
        self,
        base_model: lgb.Booster,
        factors: list[dict],
        update_start: date,
        update_end: date,
        num_boost_round: int = 50,
    ) -> lgb.Booster | None:
        """
        對模型進行增量更新

        Args:
            base_model: 原始 LightGBM 模型
            factors: 因子列表
            update_start: 更新資料開始日期
            update_end: 更新資料結束日期
            num_boost_round: 增量訓練的迭代次數

        Returns:
            更新後的模型，如果資料不足則返回 None
        """
        data = self._load_data(factors, update_start, update_end)
        if data is None:
            return None

        X, y = data

        # 建立 LightGBM Dataset
        train_data = lgb.Dataset(X.values, label=y.values)

        # 增量學習參數（較溫和的學習率）
        params = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,  # 較小的學習率避免遺忘
            "verbosity": -1,
            "seed": 42,
        }

        # 使用 init_model 進行增量更新
        updated_model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            init_model=base_model,
            keep_training_booster=True,
        )

        return updated_model

    def update_to_date(
        self,
        base_model: lgb.Booster,
        factors: list[dict],
        model_train_end: date,
        target_date: date,
        num_boost_round: int = 50,
    ) -> tuple[lgb.Booster, int] | None:
        """
        將模型增量更新到指定日期

        Args:
            base_model: 原始模型
            factors: 因子列表
            model_train_end: 模型原本的訓練結束日期
            target_date: 目標更新日期（通常是預測日期的前一天）
            num_boost_round: 增量訓練的迭代次數

        Returns:
            (updated_model, days_updated) 或 None
        """
        # 計算更新資料範圍
        # 注意：label 定義為 Ref($close, -2) / Ref($close, -1) - 1
        # 所以最新可用的 label 是 target_date - 2 天
        from datetime import timedelta

        # 更新資料從訓練結束後一天開始
        update_start = model_train_end + timedelta(days=1)

        # 考慮 label 的延遲（需要 T+2 收盤價）
        # 如果 target_date 是預測日，最新可用 label 是 target_date - 2
        update_end = target_date - timedelta(days=2)

        # 檢查是否有新資料可用
        if update_end <= update_start:
            return None

        updated_model = self.update(
            base_model=base_model,
            factors=factors,
            update_start=update_start,
            update_end=update_end,
            num_boost_round=num_boost_round,
        )

        if updated_model is None:
            return None

        days_updated = (update_end - update_start).days + 1
        return updated_model, days_updated
