"""因子選擇模組

支援兩種文獻支持的方法：
1. none: Qlib 標準流程，不做選擇，依賴 LightGBM 內建機制
2. dedup: RD-Agent IC 去重複（推薦），移除高相關因子

參考文獻：
- RD-Agent (Microsoft Research, 2025): https://arxiv.org/html/2505.15155v2
- Qlib (Microsoft): https://github.com/microsoft/qlib
"""

from src.services.factor_selection.base import (
    FactorSelectionResult,
    FactorSelector,
)
from src.services.factor_selection.ic_dedup import ICDeduplicator
from src.services.factor_selection.robust import RobustFactorSelector

__all__ = [
    "FactorSelectionResult",
    "FactorSelector",
    "ICDeduplicator",
    "RobustFactorSelector",
]
