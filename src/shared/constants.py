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

# === CPCV 參數 ===

CPCV_N_FOLDS = 6  # 分割數
CPCV_N_TEST_FOLDS = 2  # 測試 fold 數 → C(6,2)=15 條路徑
CPCV_PURGE_DAYS = 5  # Purging 天數（避免 label 洩漏）
CPCV_EMBARGO_DAYS = 5  # Embargo 天數（避免序列相關）

# === CPCV 選擇參數（MDI 方法）===
# 參考：López de Prado (2018) "Advances in Financial Machine Learning"
# 方法："Features with no gain should be excluded" → importance > 0

CPCV_TIME_DECAY_RATE = 0.95  # 時間衰減率（每 fold 衰減 5%）

# === CPCV 選擇門檻 ===

CPCV_MIN_POSITIVE_RATIO = 0.5  # 最低正向路徑比例（至少 50% 路徑有正向貢獻）
CPCV_FALLBACK_POSITIVE_RATIO = 0.4  # Fallback positive ratio 門檻
CPCV_FALLBACK_MAX_FACTORS = 30  # Fallback 最多選擇的因子數
CPCV_MIN_FACTORS_BEFORE_FALLBACK = 5  # 低於此數量時觸發 Fallback 補充

# === Bootstrap 穩定性過濾 ===

BOOTSTRAP_N_ITERATIONS = 30  # 迭代次數（精簡版）
BOOTSTRAP_STABILITY_THRESHOLD = 0.75  # 穩定性閾值（≥75% 被選中才保留）
BOOTSTRAP_SAMPLE_RATIO = 0.8  # 每次抽樣比例

# === 訓練品質監控 ===

QUALITY_JACCARD_MIN = 0.3  # Jaccard 相似度最低閾值
QUALITY_IC_STD_MAX = 0.1  # IC 標準差最高閾值
QUALITY_ICIR_MIN = 0.5  # ICIR 最低閾值
