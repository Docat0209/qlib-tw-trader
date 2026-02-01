"""
預測服務 - 使用已訓練模型預測指定日期的股票

重要：避免 Lookahead Bias
- 模型訓練時，label = Ref($close, -2) / Ref($close, -1) - 1
- 即：T 日特徵預測 T+1→T+2 收益率
- 若要在 T 日開盤交易，需使用 T-1 日特徵
"""

import json
import pickle
from datetime import date, timedelta
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
        trade_date: date,
        top_k: int = 10,
    ) -> tuple[date, list[dict]]:
        """
        預測指定交易日的 Top K 股票

        重要：避免 Lookahead Bias
        - trade_date 是預計交易的日期（買入日）
        - 系統會自動使用 trade_date 前一個交易日的資料進行預測
        - 因為只有前一日收盤後才能取得完整的特徵資料

        Args:
            model_name: 模型名稱 (YYYYMM-hash 格式)
            trade_date: 預計交易日期（買入日）
            top_k: 返回前 K 名股票

        Returns:
            (特徵資料日期, [{"symbol": ..., "score": ..., "rank": ...}])
        """
        self._init_qlib()
        from qlib.data import D

        model, factors, config = self._load_model(model_name)

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found")

        fields = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        # 查詢 trade_date 前一日的資料（避免 lookahead bias）
        # 使用範圍查詢以處理假日情況，取最後一個交易日
        lookback_start = trade_date - timedelta(days=10)  # 往前看 10 天處理連假
        lookback_end = trade_date - timedelta(days=1)  # 前一天

        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time=lookback_start.strftime("%Y-%m-%d"),
            end_time=lookback_end.strftime("%Y-%m-%d"),
        )

        if df.empty:
            raise ValueError(f"No data available before {trade_date}")

        # 取最後一個交易日的資料
        last_date = df.index.get_level_values("datetime").max()
        df = df.loc[df.index.get_level_values("datetime") == last_date]

        df.columns = names

        # 取得實際特徵資料日期
        feature_date = df.index.get_level_values("datetime")[0]
        if hasattr(feature_date, "date"):
            feature_date = feature_date.date()

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

        # 排序取 Top K（使用 symbol 作為 tie-breaker 確保穩定排序）
        result_df = result_df.sort_values(
            by=["score", "symbol"],
            ascending=[False, True],
        ).head(top_k)
        result_df["rank"] = range(1, len(result_df) + 1)

        signals = [
            {
                "symbol": row["symbol"],
                "score": float(row["score"]),
                "rank": int(row["rank"]),
            }
            for _, row in result_df.iterrows()
        ]

        return feature_date, signals
