"""
回測服務 - 使用 backtrader 進行回測
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import backtrader as bt
import numpy as np
import pandas as pd

# 台股手續費
COST_BUY = 0.001425  # 買入手續費 0.1425%
COST_SELL = 0.001425 + 0.003  # 賣出手續費 0.1425% + 證交稅 0.3%

# 模型目錄
MODELS_DIR = Path("data/models")


@dataclass
class TradeRecord:
    """交易記錄"""

    date: date
    stock_id: str
    side: str  # buy / sell
    shares: int
    price: float
    amount: float
    commission: float


@dataclass
class EquityPoint:
    """權益曲線點"""

    date: str
    equity: float
    benchmark: float | None = None
    drawdown: float | None = None


@dataclass
class BacktestMetrics:
    """回測績效指標"""

    # 核心指標
    total_return_with_cost: float
    total_return_without_cost: float
    annual_return_with_cost: float
    annual_return_without_cost: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_cost: float

    # 市場基準
    market_return: float | None = None
    market_hit_rate: float | None = None
    market_stocks_up: int | None = None
    market_stocks_down: int | None = None

    # 超額表現
    excess_return: float | None = None
    excess_hit_rate: float | None = None
    alpha: float | None = None

    # 風險調整指標
    sortino_ratio: float | None = None
    information_ratio: float | None = None
    calmar_ratio: float | None = None


@dataclass
class BacktestResult:
    """回測結果"""

    metrics: BacktestMetrics
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)


class TopKStrategy(bt.Strategy):
    """Top-K 持股策略（使用 backtrader）"""

    params = (
        ("topk", 10),
        ("scores", None),  # DataFrame: index=date, columns=stock_id
        ("trade_price", "close"),  # "close" | "open"
    )

    def __init__(self):
        self.order_dict = {}
        self.trade_records = []  # 儲存交易記錄

    def notify_order(self, order):
        """訂單執行完成時記錄"""
        if order.status != order.Completed:
            return

        stock_id = order.data._name
        dt = bt.num2date(order.executed.dt).date()
        price = float(order.executed.price)
        size = int(abs(order.executed.size))
        amount = float(price * size)

        # 計算手續費
        if order.isbuy():
            commission = amount * COST_BUY
            side = "buy"
        else:
            commission = amount * COST_SELL
            side = "sell"

        self.trade_records.append({
            "date": dt,
            "stock_id": stock_id,
            "side": side,
            "shares": size,
            "price": price,
            "amount": amount,
            "commission": commission,
        })

    def notify_trade(self, trade):
        """交易完成時的回調（現在由 notify_order 處理記錄）"""
        pass

    def next(self):
        # 取得當日日期（轉換為 pandas Timestamp 以匹配 scores index）
        current_dt = self.datas[0].datetime.datetime(0)
        current_date = pd.Timestamp(current_dt).normalize()

        # 取得前一日分數（避免 lookahead bias）
        # 分數是基於 T 日特徵預測 T+1→T+2 收益率
        # 所以在 T 日交易時，應使用 T-1 日的分數
        if self.params.scores is None:
            return

        # 找前一個交易日的分數
        available_dates = self.params.scores.index[self.params.scores.index < current_date]
        if len(available_dates) == 0:
            return

        prev_date = available_dates[-1]
        scores = self.params.scores.loc[prev_date].dropna()
        if scores.empty:
            return

        # 選擇 Top-K（使用 symbol 作為 tie-breaker 確保穩定排序）
        scores_df = scores.reset_index()
        scores_df.columns = ["symbol", "score"]
        scores_df = scores_df.sort_values(
            by=["score", "symbol"],
            ascending=[False, True],
        ).head(self.params.topk)
        topk_stocks = scores_df["symbol"].tolist()

        # 計算每檔權重
        weight = 1.0 / self.params.topk

        # 賣出不在 Top-K 的持股
        for data in self.datas:
            stock_id = data._name
            position = self.getposition(data)

            if position.size > 0 and stock_id not in topk_stocks:
                self.close(data)

        # 買入 Top-K
        for data in self.datas:
            stock_id = data._name
            if stock_id in topk_stocks:
                position = self.getposition(data)
                if position.size == 0:
                    # 計算可買股數（不限整張，方便回測）
                    target_value = self.broker.getvalue() * weight
                    # 根據 trade_price 參數選擇價格
                    if self.params.trade_price == "open":
                        price = data.open[0]
                    else:
                        price = data.close[0]
                    # 檢查價格有效性（避免 NaN 導致錯誤）
                    if price > 0 and not pd.isna(price):
                        size = int(target_value / price)
                        if size >= 1:
                            self.buy(data, size=size)


class TaiwanCommission(bt.CommInfoBase):
    """台股手續費"""

    params = (
        ("commission", COST_BUY),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 買入
            return abs(size) * price * COST_BUY
        else:  # 賣出
            return abs(size) * price * COST_SELL


class Backtester:
    """回測服務"""

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False

    def _init_qlib(self, force: bool = False) -> None:
        """初始化 qlib"""
        if self._qlib_initialized and not force:
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
            raise RuntimeError("qlib is not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def _get_instruments(self) -> list[str]:
        """取得股票清單"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        return []

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

    def _calculate_market_benchmark(
        self,
        price_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> dict:
        """
        計算市場基準指標

        使用所有股票的期間報酬計算：
        - market_return: 所有股票報酬的平均值（等權重持有）
        - market_hit_rate: 上漲股票比例（隨機選股的預期勝率）
        - market_stocks_up / market_stocks_down: 上漲/下跌股票數
        """
        # 取得每檔股票的期間收盤價
        close_df = price_df["close"].unstack(level="instrument")

        # 取得期間首尾日的價格
        available_dates = close_df.index.tolist()
        if len(available_dates) < 2:
            return {}

        # 找到最接近 start_date 和 end_date 的實際交易日
        start_prices = close_df.iloc[0]
        end_prices = close_df.iloc[-1]

        # 計算每檔股票的期間報酬
        stock_returns = (end_prices - start_prices) / start_prices
        stock_returns = stock_returns.dropna()

        if stock_returns.empty:
            return {}

        # 市場平均報酬
        market_return = stock_returns.mean() * 100

        # 上漲/下跌統計
        stocks_up = (stock_returns > 0).sum()
        stocks_down = (stock_returns <= 0).sum()
        total_stocks = stocks_up + stocks_down

        # 市場基準勝率（隨機選股的預期勝率）
        market_hit_rate = (stocks_up / total_stocks * 100) if total_stocks > 0 else 0

        return {
            "market_return": round(float(market_return), 2),
            "market_hit_rate": round(float(market_hit_rate), 1),
            "market_stocks_up": int(stocks_up),
            "market_stocks_down": int(stocks_down),
        }

    def _calculate_excess_metrics(
        self,
        model_return: float,
        model_win_rate: float,
        market_return: float | None,
        market_hit_rate: float | None,
    ) -> dict:
        """計算超額表現指標"""
        if market_return is None or market_hit_rate is None:
            return {}

        excess_return = model_return - market_return
        excess_hit_rate = model_win_rate - market_hit_rate

        return {
            "excess_return": round(excess_return, 2),
            "excess_hit_rate": round(excess_hit_rate, 1),
            "alpha": round(excess_return, 2),  # 單期 alpha 等同 excess_return
        }

    def _calculate_risk_adjusted_metrics(
        self,
        annual_return: float,
        max_drawdown: float,
        returns_analysis: dict,
        market_returns: pd.Series | None = None,
    ) -> dict:
        """
        計算風險調整指標

        - Sortino Ratio: 只考慮下行風險
        - Calmar Ratio: 報酬除以最大回撤
        - Information Ratio: alpha / tracking_error（需要市場報酬序列）
        """
        result = {}

        # 提取每日報酬序列
        daily_returns = [r for r in returns_analysis.values() if r is not None]

        if daily_returns:
            returns_arr = np.array(daily_returns)

            # Sortino Ratio: excess_return / downside_std
            # 只計算負報酬的標準差
            negative_returns = returns_arr[returns_arr < 0]
            if len(negative_returns) > 1:
                downside_std = np.std(negative_returns, ddof=1)
                if downside_std > 1e-8:
                    # 使用年化報酬
                    sortino = annual_return / 100 / (downside_std * np.sqrt(252))
                    result["sortino_ratio"] = round(float(sortino), 2)

        # Calmar Ratio: annual_return / max_drawdown
        if max_drawdown > 0:
            calmar = annual_return / max_drawdown
            result["calmar_ratio"] = round(float(calmar), 2)

        # Information Ratio（如果有市場報酬序列）
        # 目前暫不計算，需要逐日市場報酬

        return result

    def generate_predictions(
        self,
        model_name: str,
        start_date: date,
        end_date: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> pd.DataFrame:
        """
        產生預測分數

        Returns:
            pd.DataFrame: index=date, columns=stock_id, values=score
        """
        self._init_qlib()
        from qlib.data import D

        if on_progress:
            on_progress(2, f"Loading model: {model_name}...")

        model, factors, config = self._load_model(model_name)

        if on_progress:
            on_progress(3, f"Model loaded with {len(factors)} factors")

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found")

        if on_progress:
            on_progress(4, f"Found {len(instruments)} stocks in universe")

        fields = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        if on_progress:
            on_progress(5, "Querying factor data from qlib...")

        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            raise ValueError("No data available for the specified period")

        df.columns = names

        if on_progress:
            on_progress(50, f"Loaded {len(df)} data points")

        if on_progress:
            on_progress(52, "Processing infinite values...")

        df = self._process_inf(df)

        if on_progress:
            on_progress(54, "Applying cross-sectional z-score normalization...")

        df = self._zscore_by_date(df)
        df = df.fillna(0)

        if on_progress:
            on_progress(56, "Running model inference...")

        predictions = model.predict(df.values)

        if on_progress:
            on_progress(58, "Formatting prediction results...")

        pred_series = pd.Series(predictions, index=df.index, name="score")

        # 轉換為 DataFrame: index=date, columns=stock_id
        pred_df = pred_series.unstack(level="instrument")
        pred_df.index = pd.to_datetime(pred_df.index.date)

        if on_progress:
            on_progress(60, f"Generated {len(pred_df)} days of predictions")

        return pred_df

    def run(
        self,
        model_name: str,
        start_date: date,
        end_date: date,
        initial_capital: float = 1_000_000.0,
        topk: int = 10,
        n_drop: int = 1,
        trade_price: str = "close",
        on_progress: Callable[[float, str], None] | None = None,
    ) -> BacktestResult:
        """
        使用 backtrader 執行回測
        """
        self._init_qlib(force=True)
        from qlib.data import D

        # 產生預測分數（0-70%）
        scores_df = self.generate_predictions(model_name, start_date, end_date, on_progress)

        if on_progress:
            on_progress(61, "Loading OHLCV price data...")

        # 取得價格資料
        instruments = self._get_instruments()

        if on_progress:
            on_progress(62, f"Fetching data for {len(instruments)} stocks...")

        price_df = D.features(
            instruments=instruments,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d"),
        )
        price_df.columns = ["open", "high", "low", "close", "volume"]

        if on_progress:
            on_progress(82, "Initializing backtrader engine...")

        # 建立 backtrader cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.addcommissioninfo(TaiwanCommission())

        if on_progress:
            on_progress(83, "Building data feeds...")

        # 為每檔股票建立 data feed
        added_count = 0
        for i, stock_id in enumerate(instruments):
            try:
                stock_data = price_df.xs(stock_id, level="instrument")
                if stock_data.empty:
                    continue

                # 轉換為 backtrader 格式
                stock_data = stock_data.reset_index()
                stock_data.columns = ["datetime", "open", "high", "low", "close", "volume"]
                stock_data["datetime"] = pd.to_datetime(stock_data["datetime"])
                stock_data = stock_data.set_index("datetime")
                stock_data["openinterest"] = 0

                # 移除含有 NaN 的列（避免 backtrader 處理 NaN 時出錯）
                stock_data = stock_data.dropna(subset=["open", "high", "low", "close"])
                if stock_data.empty:
                    continue

                data = bt.feeds.PandasData(
                    dataname=stock_data,
                    name=stock_id,
                )
                cerebro.adddata(data)
                added_count += 1
            except (KeyError, ValueError):
                continue

            # 更新進度（83-88%）
            if on_progress and i % 10 == 0:
                progress = 83 + (i / len(instruments)) * 5
                on_progress(progress, f"Loading {stock_id} ({added_count}/{len(instruments)})...")

        if on_progress:
            on_progress(88, f"Loaded {added_count} stocks, configuring strategy...")

        # 加入策略
        cerebro.addstrategy(TopKStrategy, topk=topk, scores=scores_df, trade_price=trade_price)

        # 加入分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")

        if on_progress:
            on_progress(89, "Executing backtest simulation...")

        # 執行回測
        results = cerebro.run()
        strat = results[0]

        if on_progress:
            on_progress(96, "Calculating performance metrics...")

        # 提取結果
        final_value = cerebro.broker.getvalue()
        total_return_with = (final_value - initial_capital) / initial_capital

        # 從實際交易記錄計算總手續費
        total_cost = sum(t['commission'] for t in strat.trade_records)

        # 計算不含手續費的報酬（加回手續費）
        total_return_without = (final_value + total_cost - initial_capital) / initial_capital

        # 從分析器提取指標
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()

        sharpe = sharpe_analysis.get("sharperatio", 0) or 0
        max_dd = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0
        total_trades = trade_analysis.get("total", {}).get("total", 0) or 0

        # 計算勝率
        won = trade_analysis.get("won", {}).get("total", 0) or 0
        lost = trade_analysis.get("lost", {}).get("total", 0) or 0
        win_rate = won / (won + lost) * 100 if (won + lost) > 0 else 0

        # 計算年化報酬
        days = (end_date - start_date).days
        years = max(days / 365.0, 0.01)
        annual_return_with_cost = round(((1 + total_return_with) ** (1 / years) - 1) * 100, 2)

        # 計算市場基準指標
        market_benchmark = self._calculate_market_benchmark(price_df, start_date, end_date)

        # 計算超額表現指標
        model_return = round(total_return_with * 100, 2)
        excess_metrics = self._calculate_excess_metrics(
            model_return=model_return,
            model_win_rate=win_rate,
            market_return=market_benchmark.get("market_return"),
            market_hit_rate=market_benchmark.get("market_hit_rate"),
        )

        # 計算風險調整指標
        risk_adjusted = self._calculate_risk_adjusted_metrics(
            annual_return=annual_return_with_cost,
            max_drawdown=max_dd,
            returns_analysis=returns_analysis,
        )

        metrics = BacktestMetrics(
            total_return_with_cost=model_return,
            total_return_without_cost=round(total_return_without * 100, 2),
            annual_return_with_cost=annual_return_with_cost,
            annual_return_without_cost=round(((1 + total_return_without) ** (1 / years) - 1) * 100, 2),
            sharpe_ratio=round(float(sharpe), 2),
            max_drawdown=round(float(max_dd), 2),
            win_rate=round(float(win_rate), 1),
            total_trades=int(total_trades),
            total_cost=round(total_cost, 2),
            # 市場基準
            market_return=market_benchmark.get("market_return"),
            market_hit_rate=market_benchmark.get("market_hit_rate"),
            market_stocks_up=market_benchmark.get("market_stocks_up"),
            market_stocks_down=market_benchmark.get("market_stocks_down"),
            # 超額表現
            excess_return=excess_metrics.get("excess_return"),
            excess_hit_rate=excess_metrics.get("excess_hit_rate"),
            alpha=excess_metrics.get("alpha"),
            # 風險調整指標
            sortino_ratio=risk_adjusted.get("sortino_ratio"),
            calmar_ratio=risk_adjusted.get("calmar_ratio"),
            information_ratio=risk_adjusted.get("information_ratio"),
        )

        # 建立權益曲線
        equity_curve = []
        equity = initial_capital
        peak = equity

        for dt, ret in returns_analysis.items():
            if ret is None:
                continue
            equity = equity * (1 + ret)
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0

            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            equity_curve.append(EquityPoint(
                date=dt_str,
                equity=round(equity, 2),
                benchmark=None,
                drawdown=round(drawdown, 2),
            ))

        # 提取交易記錄
        trades = [
            TradeRecord(
                date=t["date"],
                stock_id=t["stock_id"],
                side=t["side"],
                shares=t["shares"],
                price=t["price"],
                amount=t["amount"],
                commission=t["commission"],
            )
            for t in strat.trade_records
        ]

        if on_progress:
            on_progress(100, "Backtest completed")

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
        )
