"""訓練相關常數"""

from zoneinfo import ZoneInfo

# === 時區 ===

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# === 訓練期設定（週訓練架構）===

TRAIN_DAYS = 252  # 訓練期：1 年
VALID_DAYS = 5    # 驗證期：1 週（5 個交易日）
EMBARGO_DAYS = 5  # Embargo：1 週（防止 label lookahead）

# === 重訓練設定 ===

RETRAIN_THRESHOLD_DAYS = 7  # 每週重訓

# === CPCV 參數 ===

CPCV_N_FOLDS = 6  # 分割數
CPCV_N_TEST_FOLDS = 2  # 測試 fold 數 → C(6,2)=15 條路徑
CPCV_PURGE_DAYS = 5  # Purging 天數（避免 label 洩漏）
CPCV_EMBARGO_DAYS = 5  # Embargo 天數（避免序列相關）
