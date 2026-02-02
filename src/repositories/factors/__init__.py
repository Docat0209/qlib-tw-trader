"""
因子定義模組

匯出所有因子定義，包含：
- Alpha158 純 K 線因子 (~109 個)
- 台股籌碼因子 (~130 個)
- 交互因子 (~55 個)
"""

from src.repositories.factors.alpha158 import ALPHA158_FACTORS
from src.repositories.factors.interaction import INTERACTION_FACTORS
from src.repositories.factors.taiwan_chips import TAIWAN_CHIPS_FACTORS

# 匯出所有因子
ALL_FACTORS = ALPHA158_FACTORS + TAIWAN_CHIPS_FACTORS + INTERACTION_FACTORS

__all__ = [
    "ALPHA158_FACTORS",
    "TAIWAN_CHIPS_FACTORS",
    "INTERACTION_FACTORS",
    "ALL_FACTORS",
]
