"""
預測服務 - 使用已訓練模型預測指定日期的股票
"""

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODELS_DIR = Path("data/models")
QLIB_DATA_DIR = Path("data/qlib")


class Predictor:
    """預測服務"""

    def __init__(self, qlib_data_dir: Path | str = QLIB_DATA_DIR):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False

    def _init_qlib(self) -> None:
        """初始化 qlib"""
        if self._qlib_initialized:
            return

        import qlib
        from qlib.config import REG_CN

        qlib.init(
            provider_uri=str(self.data_dir),
            region=REG_CN,
        )
        self._qlib_initialized = True

    def _load_model(self, model_name: str) -> tuple[Any, list[dict], dict]:
        """載入模型檔案"""
        model_dir = MODELS_DIR / model_name

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        factors_path = model_dir / "factors.json"
        with open(factors_path) as f:
            factors = json.load(f)

        config_path = model_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        return model, factors, config

    def _get_instruments(self) -> list[str]:
        """取得股票清單"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

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

    def predict(
        self,
        model_name: str,
        target_date: date,
        top_k: int = 10,
    ) -> tuple[date, list[dict]]:
        """
        預測指定日期的 Top K 股票

        Args:
            model_name: 模型名稱 (YYYYMM-hash 格式)
            target_date: 預測目標日期
            top_k: 返回前 K 名股票

        Returns:
            (實際預測日期, [{"symbol": ..., "score": ..., "rank": ...}])
        """
        self._init_qlib()
        from qlib.data import D

        model, factors, config = self._load_model(model_name)

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found")

        fields = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        # 查詢目標日期的資料
        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time=target_date.strftime("%Y-%m-%d"),
            end_time=target_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            raise ValueError(f"No data available for {target_date}")

        df.columns = names

        # 取得實際資料日期
        actual_date = df.index.get_level_values("datetime")[0]
        if hasattr(actual_date, "date"):
            actual_date = actual_date.date()

        # 處理資料
        df = self._process_inf(df)

        # 每日截面 z-score 標準化
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 1e-8:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0

        df = df.fillna(0)

        # 執行預測
        predictions = model.predict(df.values)

        # 建立結果
        result_df = pd.DataFrame({
            "symbol": df.index.get_level_values("instrument").tolist(),
            "score": predictions,
        })

        # 排序取 Top K
        result_df = result_df.nlargest(top_k, "score")
        result_df["rank"] = range(1, len(result_df) + 1)

        signals = [
            {
                "symbol": row["symbol"],
                "score": float(row["score"]),
                "rank": int(row["rank"]),
            }
            for _, row in result_df.iterrows()
        ]

        return actual_date, signals
