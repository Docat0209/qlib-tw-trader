"""
動態驗證期計算器

使用 Ornstein-Uhlenbeck 過程計算因子 IC 的 Half-Life，
確定最佳驗證期長度。

理論基礎：
- IC 是均值回歸的時間序列
- Half-Life 告訴我們「因子有效性持續多久」
- 驗證期應該 <= Half-Life，否則驗證的是「過期」的預測能力

參考論文：
- Ernie Chan: Half-Life of Mean Reversion
- https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor

logger = logging.getLogger(__name__)


@dataclass
class HalfLifeResult:
    """Half-Life 計算結果"""

    factor_name: str
    half_life_days: float | None
    lambda_value: float | None
    is_mean_reverting: bool
    n_observations: int
    r_squared: float | None


@dataclass
class ValidPeriodResult:
    """驗證期計算結果"""

    optimal_valid_days: int
    median_half_life: float
    factor_half_lives: dict[str, float]
    mean_reverting_count: int
    total_factors: int
    computed_at: date


class ValidPeriodCalculator:
    """
    動態驗證期計算器

    使用 Ornstein-Uhlenbeck 過程計算因子 IC 的 Half-Life，
    然後根據中位數 Half-Life 計算最佳驗證期長度。
    """

    # 驗證期限制
    MIN_VALID_DAYS = 10  # 最少 10 天（統計顯著性）
    MAX_VALID_DAYS = 60  # 最多 60 天（避免數據不足）

    # 計算所需的最小觀察天數
    MIN_OBSERVATIONS = 60

    # Half-Life 乘數（保守估計）
    HALF_LIFE_MULTIPLIER = 2.0

    def __init__(self, session: Session | None = None):
        self._session = session

    def _get_instruments(self, data_dir) -> list[str]:
        """從 qlib instruments 目錄讀取股票清單"""
        from pathlib import Path

        data_dir = Path(data_dir)
        instruments_file = data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        # 備選：從 features 目錄取得
        features_dir = data_dir / "features"
        if features_dir.exists():
            return [d.name for d in features_dir.iterdir() if d.is_dir()]

        return []

    def compute_half_life(self, ic_series: pd.Series) -> HalfLifeResult:
        """
        計算單一因子 IC 的 Half-Life

        使用 Ornstein-Uhlenbeck 過程：
        IC(t) - IC(t-1) = α + β * IC(t-1) + ε
        λ = 1 + β
        Half-Life = -ln(2) / ln(λ)

        Args:
            ic_series: 因子的每日 IC 序列

        Returns:
            HalfLifeResult
        """
        factor_name = ic_series.name if hasattr(ic_series, "name") else "unknown"

        # 移除 NaN
        ic_clean = ic_series.dropna()

        if len(ic_clean) < self.MIN_OBSERVATIONS:
            return HalfLifeResult(
                factor_name=str(factor_name),
                half_life_days=None,
                lambda_value=None,
                is_mean_reverting=False,
                n_observations=len(ic_clean),
                r_squared=None,
            )

        # 計算 ΔIC 和 lagged IC
        delta_ic = ic_clean.diff().dropna()
        lagged_ic = ic_clean.shift(1).dropna()

        # 對齊索引
        common_idx = delta_ic.index.intersection(lagged_ic.index)
        y = delta_ic.loc[common_idx].values
        x = lagged_ic.loc[common_idx].values

        if len(x) < 30:  # 回歸至少需要 30 個數據點
            return HalfLifeResult(
                factor_name=str(factor_name),
                half_life_days=None,
                lambda_value=None,
                is_mean_reverting=False,
                n_observations=len(ic_clean),
                r_squared=None,
            )

        # OLS 回歸
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # λ = 1 + slope
        lambda_val = 1 + slope

        # 檢查是否均值回歸（0 < λ < 1）
        is_mean_reverting = 0 < lambda_val < 1

        if is_mean_reverting:
            # Half-Life = -ln(2) / ln(λ)
            half_life = -np.log(2) / np.log(lambda_val)
        else:
            half_life = None

        return HalfLifeResult(
            factor_name=str(factor_name),
            half_life_days=half_life,
            lambda_value=float(lambda_val),
            is_mean_reverting=is_mean_reverting,
            n_observations=len(ic_clean),
            r_squared=float(r_value**2),
        )

    def compute_optimal_valid_days(
        self,
        factor_daily_ics: pd.DataFrame,
    ) -> ValidPeriodResult:
        """
        計算最佳驗證期長度

        Args:
            factor_daily_ics: DataFrame, index=date, columns=factor_name, values=daily_ic

        Returns:
            ValidPeriodResult
        """
        half_lives = []
        factor_half_lives = {}

        for factor_name in factor_daily_ics.columns:
            ic_series = factor_daily_ics[factor_name]
            ic_series.name = factor_name

            result = self.compute_half_life(ic_series)

            if result.is_mean_reverting and result.half_life_days is not None:
                # 過濾極端值（< 1 天或 > 200 天）
                if 1 < result.half_life_days < 200:
                    half_lives.append(result.half_life_days)
                    factor_half_lives[factor_name] = result.half_life_days

        if not half_lives:
            logger.warning("No mean-reverting factors found, using default valid days")
            return ValidPeriodResult(
                optimal_valid_days=20,  # 預設值
                median_half_life=0,
                factor_half_lives={},
                mean_reverting_count=0,
                total_factors=len(factor_daily_ics.columns),
                computed_at=date.today(),
            )

        # 取中位數
        median_half_life = float(np.median(half_lives))

        # 最佳驗證期 = Half-Life × 乘數
        suggested_days = int(median_half_life * self.HALF_LIFE_MULTIPLIER)

        # 限制在合理範圍內
        optimal_days = max(self.MIN_VALID_DAYS, min(self.MAX_VALID_DAYS, suggested_days))

        logger.info(
            f"Computed optimal valid days: {optimal_days} "
            f"(median half-life: {median_half_life:.1f}, "
            f"mean-reverting factors: {len(half_lives)}/{len(factor_daily_ics.columns)})"
        )

        return ValidPeriodResult(
            optimal_valid_days=optimal_days,
            median_half_life=median_half_life,
            factor_half_lives=factor_half_lives,
            mean_reverting_count=len(half_lives),
            total_factors=len(factor_daily_ics.columns),
            computed_at=date.today(),
        )

    def compute_from_database(
        self,
        train_start: date,
        train_end: date,
    ) -> ValidPeriodResult:
        """
        從資料庫計算最佳驗證期

        這個方法會：
        1. 載入所有啟用的因子
        2. 計算每個因子在訓練期的每日 IC
        3. 計算最佳驗證期

        Args:
            train_start: 訓練開始日期
            train_end: 訓練結束日期

        Returns:
            ValidPeriodResult
        """
        if self._session is None:
            raise ValueError("Session is required for database computation")

        import qlib
        from qlib.data import D
        from pathlib import Path

        # 取得啟用的因子
        factor_repo = FactorRepository(self._session)
        factors = factor_repo.get_enabled()

        if not factors:
            raise ValueError("No enabled factors found")

        logger.info(f"Computing optimal valid days from {len(factors)} factors")

        # 初始化 qlib
        data_dir = Path("data/qlib")
        qlib.init(provider_uri=str(data_dir))

        # 讀取 instruments 列表
        instruments = self._get_instruments(data_dir)
        if not instruments:
            raise ValueError("No instruments found in qlib data directory")

        logger.info(f"Found {len(instruments)} instruments")

        # 構建因子表達式
        factor_exprs = {f.name: f.expression for f in factors}
        label_expr = "Ref($close, -2) / Ref($close, -1) - 1"

        # 載入資料
        all_exprs = {**factor_exprs, "label": label_expr}
        data = D.features(
            instruments=instruments,
            fields=list(all_exprs.values()),
            start_time=str(train_start),
            end_time=str(train_end),
        )

        # 重命名欄位
        data.columns = list(all_exprs.keys())

        # 計算每日截面 IC
        def calc_daily_ic(group: pd.DataFrame, factor_name: str) -> float | None:
            if len(group) < 5:
                return None
            valid = group[[factor_name, "label"]].dropna()
            if len(valid) < 5:
                return None
            return float(valid[factor_name].corr(valid["label"]))

        # 對每個因子計算每日 IC
        daily_ics = {}
        for factor_name in factor_exprs.keys():
            ics = []
            dates = []
            for dt, group in data.groupby(level="datetime"):
                ic = calc_daily_ic(group, factor_name)
                if ic is not None:
                    ics.append(ic)
                    dates.append(dt)
            if ics:
                daily_ics[factor_name] = pd.Series(ics, index=dates)

        # 轉換為 DataFrame
        factor_daily_ics = pd.DataFrame(daily_ics)

        return self.compute_optimal_valid_days(factor_daily_ics)


def main():
    """命令列入口點，用於一次性計算最佳驗證期"""
    import argparse
    from datetime import timedelta

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    parser = argparse.ArgumentParser(description="計算最佳驗證期長度")
    parser.add_argument(
        "--train-days",
        type=int,
        default=252,
        help="訓練期天數（預設 252）",
    )
    args = parser.parse_args()

    # 建立資料庫連線
    engine = create_engine("sqlite:///data/data.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 計算訓練期（使用最近一年的資料）
        train_end = date.today() - timedelta(days=7)  # 預留一週
        train_start = train_end - timedelta(days=args.train_days)

        calculator = ValidPeriodCalculator(session)
        result = calculator.compute_from_database(train_start, train_end)

        print("\n" + "=" * 60)
        print("最佳驗證期計算結果")
        print("=" * 60)
        print(f"最佳驗證期: {result.optimal_valid_days} 天")
        print(f"中位數 Half-Life: {result.median_half_life:.1f} 天")
        print(f"均值回歸因子數: {result.mean_reverting_count}/{result.total_factors}")
        print(f"計算日期: {result.computed_at}")

        print("\n前 10 個 Half-Life 最短的因子:")
        sorted_factors = sorted(
            result.factor_half_lives.items(), key=lambda x: x[1]
        )[:10]
        for name, hl in sorted_factors:
            print(f"  {name}: {hl:.1f} 天")

        print("\n" + "=" * 60)
        print("建議更新 constants.py:")
        print(f"OPTIMAL_VALID_DAYS = {result.optimal_valid_days}")
        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()
