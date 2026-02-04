"""因子選擇模組

兩階段穩健因子選擇流程：
1. LightGBM 特徵重要性預篩選（與最終模型一致的非線性方法）
2. CPCV 多路徑驗證（解決單一驗證期過擬合，內含 permutation importance）
"""

from src.services.factor_selection.base import (
    FactorSelectionResult,
    FactorSelector,
)
from src.services.factor_selection.cpcv import CPCVSelector
from src.services.factor_selection.robust import RobustFactorSelector

__all__ = [
    "FactorSelectionResult",
    "FactorSelector",
    "CPCVSelector",
    "RobustFactorSelector",
]
