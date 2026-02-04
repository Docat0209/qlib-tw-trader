"""IC-based Factor Deduplication

基於 RD-Agent 論文 (Microsoft Research, 2025):
"New factors with IC_max(n) ≥ 0.99 are deemed redundant and excluded."

參考：https://arxiv.org/html/2505.15155v2

算法：
1. 計算因子間的相關係數矩陣
2. 計算每個因子對 label 的單因子 IC
3. 按 IC 降序排列，優先保留高 IC 因子
4. 對每個因子，檢查與已保留因子的相關性
5. 若相關性 >= threshold，則標記為冗餘並移除
"""

import logging

import pandas as pd

from src.repositories.models import Factor
from src.shared.constants import IC_DEDUP_THRESHOLD

logger = logging.getLogger(__name__)


class ICDeduplicator:
    """
    RD-Agent 風格的 IC 去重複

    只移除高度冗餘的因子（相關係數 >= threshold），
    不做 forward selection，保留所有非冗餘因子。
    """

    def __init__(self, correlation_threshold: float = IC_DEDUP_THRESHOLD):
        """
        Args:
            correlation_threshold: 相關係數閾值
                - RD-Agent 使用 0.99（只移除極度冗餘）
                - 可調低至 0.95 以更積極去重
        """
        self.threshold = correlation_threshold

    def deduplicate(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[Factor], dict]:
        """
        執行因子去重複

        算法（來自 RD-Agent 論文）：
        1. 計算因子間相關矩陣
        2. 計算單因子 IC（用於決定保留誰）
        3. 按 IC 降序處理，優先保留高 IC 因子
        4. 若因子與已保留因子相關性 >= threshold，則移除

        Args:
            factors: 候選因子列表
            X: 特徵資料（DataFrame，columns 為因子名稱）
            y: 標籤資料

        Returns:
            (保留的因子列表, 統計資訊)
        """
        if len(factors) == 0:
            return [], {
                "method": "rd_agent_ic_dedup",
                "input_count": 0,
                "output_count": 0,
                "removed_count": 0,
                "correlation_threshold": self.threshold,
            }

        factor_names = [f.name for f in factors]

        # 確保所有因子都在 X 中
        available_names = [n for n in factor_names if n in X.columns]
        if len(available_names) < len(factor_names):
            missing = set(factor_names) - set(available_names)
            logger.warning(f"IC Dedup: {len(missing)} factors missing from X: {list(missing)[:5]}...")

        # 只處理可用的因子
        factors_to_process = [f for f in factors if f.name in available_names]

        # 1. 計算因子間相關矩陣
        # RD-Agent: IC_{m,n}(t) = corr(F_m(t), F_n(t))
        logger.info(f"IC Dedup: Computing correlation matrix for {len(factors_to_process)} factors...")
        corr_matrix = X[available_names].corr(method="spearman")

        # 2. 計算每個因子對 label 的單因子 IC
        # 用於決定相關對中保留誰
        single_ics = {}
        for f in factors_to_process:
            try:
                ic = X[f.name].corr(y, method="spearman")
                single_ics[f.name] = ic if pd.notna(ic) else 0.0
            except Exception:
                single_ics[f.name] = 0.0

        # 3. 按單因子 IC 絕對值降序排列
        # 優先保留高 IC 因子
        sorted_factors = sorted(
            factors_to_process,
            key=lambda f: abs(single_ics.get(f.name, 0.0)),
            reverse=True,
        )

        # 4. 貪婪去重複
        kept: list[Factor] = []
        removed: set[str] = set()

        for factor in sorted_factors:
            if factor.name in removed:
                continue

            # 檢查與已保留因子的相關性
            is_redundant = False
            for kept_factor in kept:
                try:
                    corr = abs(corr_matrix.loc[factor.name, kept_factor.name])
                    if corr >= self.threshold:
                        is_redundant = True
                        removed.add(factor.name)
                        break
                except KeyError:
                    continue

            if not is_redundant:
                kept.append(factor)

        # 統計資訊
        stats = {
            "method": "rd_agent_ic_dedup",
            "input_count": len(factors),
            "output_count": len(kept),
            "removed_count": len(removed),
            "correlation_threshold": self.threshold,
            "removed_factors": list(removed)[:20],  # 只記錄前 20 個
            "top_kept_factors": [
                {"name": f.name, "ic": single_ics.get(f.name, 0.0)}
                for f in kept[:10]
            ],
        }

        logger.info(
            f"IC Dedup: {len(factors)} → {len(kept)} factors "
            f"(removed {len(removed)} with corr >= {self.threshold})"
        )

        return kept, stats
