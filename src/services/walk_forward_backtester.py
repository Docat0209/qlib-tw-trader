"""
Walk-Forward 回測服務

多模型依序回測，計算 Live IC 和收益
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from src.repositories.models import TrainingRun
from src.repositories.training import TrainingRepository
from src.services.incremental_learner import IncrementalLearner
from src.shared.week_utils import (
    compare_week_ids,
    get_current_week_id,
    get_next_week_id,
    get_previous_week_id,
    get_week_valid_end,
    get_weeks_in_range,
    parse_week_id,
)

# 模型目錄
MODELS_DIR = Path("data/models")
QLIB_DATA_DIR = Path("data/qlib")


@dataclass
class WeekModelInfo:
    """週模型資訊"""

    predict_week: str  # 預測的週
    model_week: str  # 使用的模型週（可能是 fallback）
    model_name: str
    model_id: int
    valid_ic: float | None
    is_fallback: bool


@dataclass
class WeekResult:
    """單週回測結果"""

    predict_week: str
    model_week: str
    model_name: str
    valid_ic: float | None
    live_ic: float | None
    ic_decay: float | None
    week_return: float | None
    market_return: float | None
    is_fallback: bool
    incremental_days: int | None = None  # 增量學習使用的天數


@dataclass
class IcAnalysis:
    """IC 分析結果"""

    avg_valid_ic: float
    avg_live_ic: float
    ic_decay: float  # (valid - live) / valid * 100
    ic_correlation: float | None


@dataclass
class ReturnMetrics:
    """收益指標"""

    cumulative_return: float
    market_return: float
    excess_return: float
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_rate: float | None
    total_trades: int


@dataclass
class EquityPoint:
    """權益曲線點"""

    date: str
    equity: float
    benchmark: float | None = None
    drawdown: float | None = None


@dataclass
class WalkForwardResult:
    """Walk-Forward 回測結果"""

    ic_analysis: IcAnalysis
    return_metrics: ReturnMetrics
    weekly_details: list[WeekResult]
    equity_curve: list[EquityPoint] = field(default_factory=list)


class WalkForwardBacktester:
    """Walk-Forward 回測服務"""

    def __init__(self, session: Session, qlib_data_dir: Path | None = None):
        self._session = session
        self._qlib_data_dir = qlib_data_dir or QLIB_DATA_DIR
        self._qlib_initialized = False

    def _init_qlib(self, force: bool = False) -> None:
        """初始化 qlib"""
        if self._qlib_initialized and not force:
            return

        try:
            import qlib
            from qlib.config import REG_CN

            qlib.init(
                provider_uri=str(self._qlib_data_dir),
                region=REG_CN,
            )
            self._qlib_initialized = True
        except ImportError:
            raise RuntimeError("qlib is not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def get_available_weeks(self) -> list[dict]:
        """
        取得可回測的週列表

        Returns:
            [
                {"week_id": "2025W01", "status": "available", "model_name": "...", "valid_ic": 0.28},
                {"week_id": "2025W05", "status": "missing", "fallback_week": "2025W04", ...},
                {"week_id": "2026W06", "status": "not_allowed", "reason": "current_week"},
            ]
        """
        training_repo = TrainingRepository(self._session)
        current_week = get_current_week_id()

        # 取得所有已訓練的模型
        trained_models = self._get_trained_models()

        # 建立週 -> 模型的映射
        week_to_model: dict[str, TrainingRun] = {}
        for model in trained_models:
            if model.week_id:
                week_to_model[model.week_id] = model

        # 取得所有週（從最早的模型週到當前週的前一週）
        if not week_to_model:
            return []

        earliest_week = min(week_to_model.keys())
        # 結束週是當前週的前兩週（因為最新一週不能回測）
        latest_allowed = get_previous_week_id(current_week)

        all_weeks = get_weeks_in_range(earliest_week, latest_allowed)

        result = []
        for week_id in all_weeks:
            # 計算「預測週」（模型週的下一週）
            predict_week = get_next_week_id(week_id)

            # 檢查預測週是否是當前週（不允許）
            if compare_week_ids(predict_week, current_week) >= 0:
                result.append({
                    "week_id": week_id,
                    "status": "not_allowed",
                    "reason": "current_week",
                })
                continue

            # 檢查該週是否有訓練好的模型
            if week_id in week_to_model:
                model = week_to_model[week_id]
                result.append({
                    "week_id": week_id,
                    "status": "available",
                    "model_name": model.name,
                    "valid_ic": float(model.model_ic) if model.model_ic else None,
                })
            else:
                # 找 fallback 模型
                fallback_week = self._find_fallback_week(week_id, week_to_model)
                if fallback_week:
                    fallback_model = week_to_model[fallback_week]
                    result.append({
                        "week_id": week_id,
                        "status": "missing",
                        "fallback_week": fallback_week,
                        "fallback_model": fallback_model.name,
                    })
                else:
                    result.append({
                        "week_id": week_id,
                        "status": "not_allowed",
                        "reason": "no_model_available",
                    })

        return result

    def _get_trained_models(self) -> list[TrainingRun]:
        """取得所有已訓練的模型（按週排序）"""
        from sqlalchemy import select

        stmt = (
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .where(TrainingRun.week_id.isnot(None))
            .order_by(TrainingRun.week_id)
        )
        return list(self._session.execute(stmt).scalars().all())

    def _find_fallback_week(
        self,
        target_week: str,
        week_to_model: dict[str, TrainingRun],
    ) -> str | None:
        """
        找到 fallback 模型週

        從 target_week 往前找最近的有模型的週
        """
        current = get_previous_week_id(target_week)
        max_lookback = 10  # 最多往前找 10 週

        for _ in range(max_lookback):
            if current in week_to_model:
                return current
            current = get_previous_week_id(current)

        return None

    def _collect_models_for_range(
        self,
        start_week_id: str,
        end_week_id: str,
    ) -> list[WeekModelInfo]:
        """
        收集回測期間需要的模型

        Args:
            start_week_id: 起始模型週（如 "2024W01"）
            end_week_id: 結束模型週（如 "2025W20"）

        Returns:
            模型資訊列表（包含 fallback 資訊）
        """
        trained_models = self._get_trained_models()
        week_to_model: dict[str, TrainingRun] = {}
        for model in trained_models:
            if model.week_id:
                week_to_model[model.week_id] = model

        all_weeks = get_weeks_in_range(start_week_id, end_week_id)
        result = []

        for week_id in all_weeks:
            predict_week = get_next_week_id(week_id)

            if week_id in week_to_model:
                model = week_to_model[week_id]
                result.append(WeekModelInfo(
                    predict_week=predict_week,
                    model_week=week_id,
                    model_name=model.name or f"m{model.id:03d}",
                    model_id=model.id,
                    valid_ic=float(model.model_ic) if model.model_ic else None,
                    is_fallback=False,
                ))
            else:
                # 使用 fallback
                fallback_week = self._find_fallback_week(week_id, week_to_model)
                if fallback_week:
                    model = week_to_model[fallback_week]
                    result.append(WeekModelInfo(
                        predict_week=predict_week,
                        model_week=fallback_week,
                        model_name=model.name or f"m{model.id:03d}",
                        model_id=model.id,
                        valid_ic=float(model.model_ic) if model.model_ic else None,
                        is_fallback=True,
                    ))
                # 如果找不到 fallback，跳過這週

        return result

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

    def _predict_week(
        self,
        model: Any,
        factors: list[dict],
        predict_start: date,
        predict_end: date,
    ) -> pd.DataFrame:
        """
        對指定週期進行預測

        Returns:
            DataFrame: index=date, columns=stock_id, values=score
        """
        self._init_qlib()
        from qlib.data import D

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found")

        fields = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time=predict_start.strftime("%Y-%m-%d"),
            end_time=predict_end.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return pd.DataFrame()

        df.columns = names

        # 處理數據
        df = self._process_inf(df)
        df = self._zscore_by_date(df)
        df = df.fillna(0)

        # 預測
        predictions = model.predict(df.values)
        pred_series = pd.Series(predictions, index=df.index, name="score")

        # 轉換為 DataFrame: index=date, columns=stock_id
        pred_df = pred_series.unstack(level="instrument")
        pred_df.index = pd.to_datetime(pred_df.index.date)

        return pred_df

    def _get_instruments(self) -> list[str]:
        """取得股票清單"""
        instruments_file = self._qlib_data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        return []

    def _calculate_live_ic(
        self,
        predictions: pd.DataFrame,
        predict_start: date,
        predict_end: date,
    ) -> float | None:
        """
        計算 Live IC

        Live IC = corr(預測分數, 實際收益)

        重要：收益計算必須對齊 Label 定義！
        - Label: Ref($close, -2) / Ref($close, -1) - 1 = T+1 收盤 → T+2 收盤
        - T 日分數預測的是 T+1→T+2 的收益

        Args:
            predictions: DataFrame with index=date, columns=stock_id, values=score
            predict_start: 預測期開始日期
            predict_end: 預測期結束日期

        Returns:
            Live IC 值
        """
        self._init_qlib()
        from qlib.data import D

        instruments = list(predictions.columns)
        if not instruments:
            return None

        # 擴展查詢範圍：往後多取 5 天（確保能計算 T+1→T+2 收益）
        extended_end = predict_end + timedelta(days=7)

        price_df = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=predict_start.strftime("%Y-%m-%d"),
            end_time=extended_end.strftime("%Y-%m-%d"),
        )

        if price_df.empty:
            return None

        price_df.columns = ["close"]

        # 計算 Label 對齊的收益：T+1 收盤 → T+2 收盤
        # 對於 T 日，計算 close[T+2] / close[T+1] - 1
        def calc_forward_returns(group: pd.DataFrame) -> pd.Series:
            close = group["close"]
            # shift(-1) 是 T+1 的價格, shift(-2) 是 T+2 的價格
            return close.shift(-2) / close.shift(-1) - 1

        returns = price_df.groupby(level="instrument", group_keys=False).apply(calc_forward_returns)
        returns = returns.dropna()

        if returns.empty:
            return None

        # 整理為寬表
        returns_wide = returns.unstack(level="instrument")
        returns_wide.index = pd.to_datetime(returns_wide.index.date)

        # 對齊日期
        common_dates = predictions.index.intersection(returns_wide.index)
        if len(common_dates) == 0:
            return None

        # 計算每日 IC
        ics = []
        for dt in common_dates:
            pred_row = predictions.loc[dt].dropna()
            ret_row = returns_wide.loc[dt].dropna()

            common_stocks = pred_row.index.intersection(ret_row.index)
            if len(common_stocks) < 10:  # 至少需要 10 個股票
                continue

            ic, _ = stats.spearmanr(
                pred_row[common_stocks].values,
                ret_row[common_stocks].values,
            )
            if not np.isnan(ic):
                ics.append(ic)

        if not ics:
            return None

        return float(np.mean(ics))

    def _get_week_date_range(self, week_id: str) -> tuple[date, date]:
        """取得週的日期範圍"""
        year, week = parse_week_id(week_id)
        monday = date.fromisocalendar(year, week, 1)
        friday = date.fromisocalendar(year, week, 5)
        return monday, friday

    def run(
        self,
        start_week_id: str,
        end_week_id: str,
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        trade_price: str = "open",
        enable_incremental: bool = False,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> WalkForwardResult:
        """
        執行 Walk-Forward 回測

        Args:
            start_week_id: 起始模型週（如 "2024W01"）
            end_week_id: 結束模型週（如 "2025W20"）
            initial_capital: 初始資金
            max_positions: 最大持倉數
            trade_price: 交易價格（"open" 或 "close"）
            enable_incremental: 是否啟用增量學習（Phase 2 實作）
            on_progress: 進度回調

        Returns:
            WalkForwardResult
        """
        from src.services.qlib_exporter import ExportConfig, QlibExporter

        if on_progress:
            on_progress(1, "Collecting models...")

        # 收集模型
        model_infos = self._collect_models_for_range(start_week_id, end_week_id)

        if not model_infos:
            raise ValueError(f"No models available for range {start_week_id} ~ {end_week_id}")

        if on_progress:
            on_progress(5, f"Found {len(model_infos)} weeks to backtest")

        # 計算整體日期範圍
        first_predict = self._get_week_date_range(model_infos[0].predict_week)[0]
        last_predict = self._get_week_date_range(model_infos[-1].predict_week)[1]

        # 匯出 qlib 資料
        lookback_days = 180
        export_start = first_predict - timedelta(days=lookback_days)

        if on_progress:
            on_progress(8, f"Exporting qlib data: {export_start} ~ {last_predict}")

        exporter = QlibExporter(self._session)
        export_config = ExportConfig(
            start_date=export_start,
            end_date=last_predict,
            output_dir=self._qlib_data_dir,
        )
        exporter.export(export_config)

        if on_progress:
            on_progress(15, "Qlib data ready, starting weekly backtests...")

        # 逐週回測
        weekly_results: list[WeekResult] = []
        all_predictions: dict[str, pd.DataFrame] = {}  # week_id -> predictions

        # 建立增量學習器（如果啟用）
        incremental_learner = None
        if enable_incremental:
            incremental_learner = IncrementalLearner(self._session)

        total_weeks = len(model_infos)
        for i, info in enumerate(model_infos):
            progress = 15 + (i / total_weeks) * 70  # 15% ~ 85%
            if on_progress:
                on_progress(progress, f"[{i+1}/{total_weeks}] Processing {info.predict_week}...")

            try:
                # 載入模型
                model, factors, config = self._load_model(info.model_name)

                # 取得預測週的日期範圍
                predict_start, predict_end = self._get_week_date_range(info.predict_week)

                # 增量學習（如果啟用）
                incremental_days = None
                if enable_incremental and incremental_learner is not None:
                    # 取得模型訓練結束日期
                    train_end_str = config.get("train_end")
                    if train_end_str:
                        model_train_end = date.fromisoformat(train_end_str)

                        # 將模型更新到預測日前一天
                        target_date = predict_start - timedelta(days=1)

                        result = incremental_learner.update_to_date(
                            base_model=model,
                            factors=factors,
                            model_train_end=model_train_end,
                            target_date=target_date,
                        )

                        if result is not None:
                            model, incremental_days = result

                # 預測
                predictions = self._predict_week(model, factors, predict_start, predict_end)

                if predictions.empty:
                    weekly_results.append(WeekResult(
                        predict_week=info.predict_week,
                        model_week=info.model_week,
                        model_name=info.model_name,
                        valid_ic=info.valid_ic,
                        live_ic=None,
                        ic_decay=None,
                        week_return=None,
                        market_return=None,
                        is_fallback=info.is_fallback,
                        incremental_days=incremental_days,
                    ))
                    continue

                all_predictions[info.predict_week] = predictions

                # 計算 Live IC
                live_ic = self._calculate_live_ic(predictions, predict_start, predict_end)

                # 計算 IC decay
                ic_decay = None
                if info.valid_ic is not None and live_ic is not None and info.valid_ic != 0:
                    ic_decay = ((info.valid_ic - live_ic) / info.valid_ic) * 100

                # 計算週收益（簡化版：Top-K 等權重）
                week_return, market_return = self._calculate_week_return(
                    predictions, predict_start, predict_end, max_positions
                )

                weekly_results.append(WeekResult(
                    predict_week=info.predict_week,
                    model_week=info.model_week,
                    model_name=info.model_name,
                    valid_ic=info.valid_ic,
                    live_ic=live_ic,
                    ic_decay=ic_decay,
                    week_return=week_return,
                    market_return=market_return,
                    is_fallback=info.is_fallback,
                    incremental_days=incremental_days,
                ))

            except Exception as e:
                # 記錄錯誤但繼續
                weekly_results.append(WeekResult(
                    predict_week=info.predict_week,
                    model_week=info.model_week,
                    model_name=info.model_name,
                    valid_ic=info.valid_ic,
                    live_ic=None,
                    ic_decay=None,
                    week_return=None,
                    market_return=None,
                    is_fallback=info.is_fallback,
                    incremental_days=None,
                ))

        if on_progress:
            on_progress(88, "Calculating summary metrics...")

        # 計算 IC 分析
        ic_analysis = self._calculate_ic_analysis(weekly_results)

        # 計算收益指標
        return_metrics = self._calculate_return_metrics(weekly_results, initial_capital)

        # 建立權益曲線
        equity_curve = self._build_equity_curve(weekly_results, initial_capital)

        if on_progress:
            on_progress(100, "Walk-forward backtest completed")

        return WalkForwardResult(
            ic_analysis=ic_analysis,
            return_metrics=return_metrics,
            weekly_details=weekly_results,
            equity_curve=equity_curve,
        )

    def _calculate_week_return(
        self,
        predictions: pd.DataFrame,
        predict_start: date,
        predict_end: date,
        max_positions: int,
    ) -> tuple[float | None, float | None]:
        """
        計算週收益

        使用 Top-K 等權重策略

        Returns:
            (week_return, market_return) in percentage
        """
        self._init_qlib()
        from qlib.data import D

        instruments = list(predictions.columns)
        if not instruments:
            return None, None

        # 取得價格資料
        price_df = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=predict_start.strftime("%Y-%m-%d"),
            end_time=predict_end.strftime("%Y-%m-%d"),
        )

        if price_df.empty:
            return None, None

        price_df.columns = ["close"]
        close_wide = price_df["close"].unstack(level="instrument")

        if close_wide.empty or len(close_wide) < 2:
            return None, None

        # 期初和期末價格
        start_prices = close_wide.iloc[0]
        end_prices = close_wide.iloc[-1]

        # 計算所有股票收益
        all_returns = (end_prices - start_prices) / start_prices
        all_returns = all_returns.dropna()

        if all_returns.empty:
            return None, None

        # 市場平均收益
        market_return = float(all_returns.mean() * 100)

        # 選擇 Top-K（使用第一天的預測分數）
        first_date = predictions.index[0]
        scores = predictions.loc[first_date].dropna()

        if scores.empty:
            return None, market_return

        # 排序選股
        scores_df = scores.reset_index()
        scores_df.columns = ["symbol", "score"]
        scores_df = scores_df.sort_values(
            by=["score", "symbol"],
            ascending=[False, True],
        ).head(max_positions)
        topk_stocks = scores_df["symbol"].tolist()

        # Top-K 收益
        topk_returns = all_returns.reindex(topk_stocks).dropna()
        if topk_returns.empty:
            return None, market_return

        week_return = float(topk_returns.mean() * 100)

        return week_return, market_return

    def _calculate_ic_analysis(self, weekly_results: list[WeekResult]) -> IcAnalysis:
        """計算 IC 分析"""
        valid_ics = [r.valid_ic for r in weekly_results if r.valid_ic is not None]
        live_ics = [r.live_ic for r in weekly_results if r.live_ic is not None]

        avg_valid_ic = float(np.mean(valid_ics)) if valid_ics else 0.0
        avg_live_ic = float(np.mean(live_ics)) if live_ics else 0.0

        # IC decay
        ic_decay = 0.0
        if avg_valid_ic != 0:
            ic_decay = ((avg_valid_ic - avg_live_ic) / avg_valid_ic) * 100

        # IC correlation
        ic_correlation = None
        paired = [
            (r.valid_ic, r.live_ic)
            for r in weekly_results
            if r.valid_ic is not None and r.live_ic is not None
        ]
        if len(paired) >= 3:
            valid_arr = [p[0] for p in paired]
            live_arr = [p[1] for p in paired]
            corr, _ = stats.pearsonr(valid_arr, live_arr)
            if not np.isnan(corr):
                ic_correlation = float(corr)

        return IcAnalysis(
            avg_valid_ic=avg_valid_ic,
            avg_live_ic=avg_live_ic,
            ic_decay=ic_decay,
            ic_correlation=ic_correlation,
        )

    def _calculate_return_metrics(
        self,
        weekly_results: list[WeekResult],
        initial_capital: float,
    ) -> ReturnMetrics:
        """計算收益指標"""
        week_returns = [r.week_return for r in weekly_results if r.week_return is not None]
        market_returns = [r.market_return for r in weekly_results if r.market_return is not None]

        if not week_returns:
            return ReturnMetrics(
                cumulative_return=0.0,
                market_return=0.0,
                excess_return=0.0,
                sharpe_ratio=None,
                max_drawdown=None,
                win_rate=None,
                total_trades=0,
            )

        # 累積收益
        cumulative = 1.0
        for ret in week_returns:
            cumulative *= (1 + ret / 100)
        cumulative_return = (cumulative - 1) * 100

        # 市場累積收益
        market_cumulative = 1.0
        for ret in market_returns:
            market_cumulative *= (1 + ret / 100)
        market_return = (market_cumulative - 1) * 100

        # 超額收益
        excess_return = cumulative_return - market_return

        # Sharpe ratio（使用週收益）
        sharpe_ratio = None
        if len(week_returns) >= 2:
            ret_arr = np.array(week_returns)
            if ret_arr.std() > 0:
                sharpe_ratio = float(ret_arr.mean() / ret_arr.std() * np.sqrt(52))  # 年化

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(week_returns)

        # Win rate（跑贏市場的週比例）
        win_weeks = sum(
            1 for r in weekly_results
            if r.week_return is not None and r.market_return is not None
            and r.week_return > r.market_return
        )
        total_valid = sum(
            1 for r in weekly_results
            if r.week_return is not None and r.market_return is not None
        )
        win_rate = (win_weeks / total_valid * 100) if total_valid > 0 else None

        return ReturnMetrics(
            cumulative_return=cumulative_return,
            market_return=market_return,
            excess_return=excess_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(weekly_results) * 10,  # 估計值
        )

    def _calculate_max_drawdown(self, weekly_returns: list[float]) -> float | None:
        """計算最大回撤"""
        if not weekly_returns:
            return None

        equity = 100.0
        peak = equity
        max_dd = 0.0

        for ret in weekly_returns:
            equity *= (1 + ret / 100)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        return float(max_dd)

    def _build_equity_curve(
        self,
        weekly_results: list[WeekResult],
        initial_capital: float,
    ) -> list[EquityPoint]:
        """建立權益曲線"""
        equity = initial_capital
        benchmark = initial_capital
        peak = equity
        curve = []

        for result in weekly_results:
            if result.week_return is not None:
                equity *= (1 + result.week_return / 100)
            if result.market_return is not None:
                benchmark *= (1 + result.market_return / 100)

            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0

            curve.append(EquityPoint(
                date=result.predict_week,
                equity=round(equity, 2),
                benchmark=round(benchmark, 2),
                drawdown=round(drawdown, 2),
            ))

        return curve
