"""訓練相關常數"""

from zoneinfo import ZoneInfo

# === 時區 ===

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# === 訓練期設定（週訓練架構）===
# 參考：機構級量化交易最佳實踐
# - 訓練期：2 年（涵蓋多個市場週期）
# - 驗證期：20% 訓練期（統計顯著性）
# - IC 標準誤目標：< 0.10

TRAIN_DAYS = 504   # 訓練期：2 年（504 個交易日）
VALID_DAYS = 100   # 驗證期：約 4 個月（20% 訓練期）
EMBARGO_DAYS = 7   # Embargo：7 天（防止 label lookahead）

# === 重訓練設定 ===

RETRAIN_THRESHOLD_DAYS = 7  # 每週重訓

# === IC Deduplication (RD-Agent) ===
# 參考：https://arxiv.org/html/2505.15155v2
# "New factors with IC_max(n) ≥ 0.99 are deemed redundant and excluded."

IC_DEDUP_THRESHOLD = 0.99  # RD-Agent 使用 0.99

# === 訓練品質監控 ===

QUALITY_JACCARD_MIN = 0.3  # Jaccard 相似度最低閾值
QUALITY_IC_STD_MAX = 0.1   # IC 標準差最高閾值
QUALITY_ICIR_MIN = 0.5     # ICIR 最低閾值
