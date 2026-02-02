"""
多期回測分析服務

計算跨多個回測期間的累積統計和統計信賴度
"""

import json
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PeriodResult:
    """單期結果"""

    period: str  # YYYYMM
    model_return: float
    market_return: float
    excess_return: float
    win_rate: float
    market_hit_rate: float
    beat_market: bool


@dataclass
class BacktestSummaryResult:
    """多期統計結果"""

    selection_method: str | None
    n_periods: int
    cumulative_return: float
    cumulative_excess_return: float
    avg_period_return: float
    avg_excess_return: float
    period_win_rate: float  # 跑贏市場期數比例
    return_std: float
    excess_return_std: float
    t_statistic: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    is_significant: bool
    periods: list[PeriodResult]


class BacktestAnalyzer:
    """多期回測分析服務"""

    def calculate_summary(
        self,
        backtests: list[dict],
        selection_method: str | None = None,
    ) -> BacktestSummaryResult | None:
        """
        聚合多期回測，計算：
        1. 累積報酬
        2. 統計信賴度
        3. 期間勝率

        Args:
            backtests: 回測記錄列表，每個需包含：
                - model_id: 模型 ID
                - start_date: 開始日期
                - end_date: 結束日期
                - result: JSON 字串（含 metrics）
            selection_method: 只計算此選擇方法的回測
        """
        if not backtests:
            return None

        periods: list[PeriodResult] = []
        model_returns: list[float] = []
        market_returns: list[float] = []
        excess_returns: list[float] = []

        for bt in backtests:
            # 解析 result JSON
            result_json = bt.get("result")
            if not result_json:
                continue

            try:
                metrics = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                continue

            # 取得必要欄位
            model_return = metrics.get("total_return_with_cost")
            market_return = metrics.get("market_return")

            if model_return is None or market_return is None:
                continue

            # 計算期間標籤 (YYYYMM)
            end_date = bt.get("end_date")
            if hasattr(end_date, "strftime"):
                period = end_date.strftime("%Y%m")
            else:
                period = str(end_date)[:7].replace("-", "")

            excess_return = model_return - market_return
            win_rate = metrics.get("win_rate", 0)
            market_hit_rate = metrics.get("market_hit_rate", 0)
            beat_market = model_return > market_return

            periods.append(PeriodResult(
                period=period,
                model_return=model_return,
                market_return=market_return,
                excess_return=excess_return,
                win_rate=win_rate,
                market_hit_rate=market_hit_rate,
                beat_market=beat_market,
            ))

            model_returns.append(model_return)
            market_returns.append(market_return)
            excess_returns.append(excess_return)

        if not periods:
            return None

        n_periods = len(periods)

        # 累積報酬（複合）
        cumulative_return = self._compound_returns(model_returns)
        cumulative_market = self._compound_returns(market_returns)
        cumulative_excess_return = cumulative_return - cumulative_market

        # 平均報酬
        avg_period_return = float(np.mean(model_returns))
        avg_excess_return = float(np.mean(excess_returns))

        # 跑贏市場期數比例
        beat_market_count = sum(1 for p in periods if p.beat_market)
        period_win_rate = (beat_market_count / n_periods * 100) if n_periods > 0 else 0

        # 標準差
        return_std = float(np.std(model_returns, ddof=1)) if n_periods > 1 else 0
        excess_return_std = float(np.std(excess_returns, ddof=1)) if n_periods > 1 else 0

        # T 檢定（檢驗超額報酬是否顯著大於 0）
        t_stat, p_value, ci_lower, ci_upper = self._calculate_t_test(excess_returns)

        # 是否統計顯著
        is_significant = p_value is not None and p_value < 0.05

        # 按期間排序
        periods.sort(key=lambda p: p.period)

        return BacktestSummaryResult(
            selection_method=selection_method,
            n_periods=n_periods,
            cumulative_return=round(cumulative_return, 2),
            cumulative_excess_return=round(cumulative_excess_return, 2),
            avg_period_return=round(avg_period_return, 2),
            avg_excess_return=round(avg_excess_return, 2),
            period_win_rate=round(period_win_rate, 1),
            return_std=round(return_std, 2),
            excess_return_std=round(excess_return_std, 2),
            t_statistic=round(t_stat, 3) if t_stat is not None else None,
            p_value=round(p_value, 4) if p_value is not None else None,
            ci_lower=round(ci_lower, 2) if ci_lower is not None else None,
            ci_upper=round(ci_upper, 2) if ci_upper is not None else None,
            is_significant=is_significant,
            periods=periods,
        )

    def _compound_returns(self, returns: list[float]) -> float:
        """
        計算複合報酬

        returns: 百分比報酬列表（如 [10.5, -2.3, 5.0]）
        """
        if not returns:
            return 0.0

        product = 1.0
        for r in returns:
            product *= (1 + r / 100)
        return (product - 1) * 100

    def _calculate_t_test(
        self,
        excess_returns: list[float],
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        計算 t 統計量和 p 值（單尾檢定）

        檢驗：超額報酬是否顯著大於 0

        Returns:
            (t_statistic, p_value, ci_lower, ci_upper)
        """
        n = len(excess_returns)
        if n < 2:
            return None, None, None, None

        returns_arr = np.array(excess_returns)
        mean = float(np.mean(returns_arr))
        std = float(np.std(returns_arr, ddof=1))

        if std < 1e-8:
            return None, None, None, None

        se = std / np.sqrt(n)
        t_stat = mean / se

        # 單尾檢定：H0: mean <= 0, H1: mean > 0
        # p_value = P(T > t_stat)
        p_value = 1 - stats.t.cdf(t_stat, n - 1)

        # 95% 信賴區間
        t_critical = stats.t.ppf(0.975, n - 1)  # 雙尾 95%
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se

        return t_stat, p_value, ci_lower, ci_upper
