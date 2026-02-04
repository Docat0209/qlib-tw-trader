"""訓練相關常數"""

from zoneinfo import ZoneInfo

# === 時區 ===

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# === 訓練期設定（週訓練架構）===

TRAIN_DAYS = 252  # 訓練期：1 年
VALID_DAYS = 5    # 驗證期：5 個交易日（避免連續週重疊）
EMBARGO_DAYS = 7  # Embargo：7 天（防止 label lookahead，對齊週移動間隔）

# === 重訓練設定 ===

RETRAIN_THRESHOLD_DAYS = 7  # 每週重訓

# === IC Deduplication (RD-Agent) ===
# 參考：https://arxiv.org/html/2505.15155v2
# "New factors with IC_max(n) ≥ 0.99 are deemed redundant and excluded."

IC_DEDUP_THRESHOLD = 0.99  # RD-Agent 使用 0.99

# === 訓練品質監控 ===

QUALITY_JACCARD_MIN = 0.3  # Jaccard 相似度最低閾值
QUALITY_IC_STD_MAX = 0.1  # IC 標準差最高閾值
QUALITY_ICIR_MIN = 0.5  # ICIR 最低閾值
