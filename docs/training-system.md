# 訓練與回測系統技術文檔

本文檔詳細說明系統的模型訓練流程、回測流程、使用的方法、模型和參數。

## 目錄

1. [系統架構概覽](#系統架構概覽)
2. [Label 定義與處理](#label-定義與處理)
3. [因子選擇流程](#因子選擇流程)
4. [模型訓練流程](#模型訓練流程)
5. [回測流程](#回測流程)
6. [IC 計算方法](#ic-計算方法)
7. [關鍵參數配置](#關鍵參數配置)
8. [參考文獻](#參考文獻)

---

## 系統架構概覽

```
┌─────────────────────────────────────────────────────────────────┐
│                        訓練流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  資料庫 ──→ QlibExporter ──→ Qlib .bin 檔案                      │
│                                    │                            │
│                                    ▼                            │
│                           D.features() 載入                      │
│                                    │                            │
│                                    ▼                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   因子選擇流程                            │   │
│  │  Bootstrap 穩定性過濾 ──→ CPCV 多路徑驗證                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                    │                            │
│                                    ▼                            │
│                           LightGBM 訓練                         │
│                                    │                            │
│                                    ▼                            │
│                     模型 + 因子清單 + 配置                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        回測流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模型載入 ──→ 預測分數 ──→ Top-K 選股 ──→ 計算收益               │
│                  │                                              │
│                  ▼                                              │
│           計算 Live IC（Spearman）                               │
│                  │                                              │
│                  ▼                                              │
│            IC Decay 分析                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Label 定義與處理

### Label 定義

```python
label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
```

**含義**：T 日的 Label = T+1 收盤價 → T+2 收盤價的收益率

**為什麼不是 T → T+1？**
- 台股 T 日收盤後，最早 T+1 開盤才能買入
- 買入後最早 T+2 才能賣出
- 因此預測 T+1 → T+2 的收益才有實際交易意義

### Label 標準化：CSRankNorm

```python
def _rank_by_date(self, series: pd.Series) -> pd.Series:
    """每日截面排名標準化"""
    def rank_pct(x: pd.Series) -> pd.Series:
        return x.rank(pct=True, method="average")
    return series.groupby(level="datetime", group_keys=False).apply(rank_pct)
```

**為什麼使用排名而非原始收益？**

| 方法 | 優點 | 缺點 |
|------|------|------|
| 原始收益 | 保留幅度信息 | 對離群值敏感、分佈偏斜 |
| CSZScoreNorm | 標準化幅度 | 仍受離群值影響 |
| **CSRankNorm** | 穩健、均勻分佈 | 失去幅度信息（但對排名策略無影響） |

**學術支持**：
- [Learning to Rank 論文](https://arxiv.org/abs/2012.07149)：預測排名比預測收益更容易，Sharpe Ratio 提升 3 倍
- [Qlib 官方配置](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml)：GRU/LSTM/AdaRNN 都使用 CSRankNorm

### 訓練與回測的一致性

| 階段 | Label 處理 | IC 計算方法 |
|------|-----------|------------|
| 訓練 | CSRankNorm（排名百分位） | Spearman（排名相關） |
| 回測 | 原始收益率 | Spearman（排名相關） |

**關鍵**：雖然 Label 處理不同，但 **Spearman 相關係數本身就是排名相關**，因此兩者計算的 IC 具有可比性。

---

## 因子選擇流程

### 兩階段選擇

```
266 個候選因子
       │
       ▼
┌─────────────────────┐
│ Bootstrap 穩定性過濾 │  ← 輕量：單因子 IC
│ (30 次迭代, ≥75%)   │
└─────────────────────┘
       │
       ▼
   ~50-80 個穩定因子
       │
       ▼
┌─────────────────────┐
│   CPCV 多路徑驗證    │  ← 嚴格：15 條路徑
│ (t≥3.0, 時間衰減)   │
└─────────────────────┘
       │
       ▼
   5-20 個最終因子
```

### Bootstrap 穩定性過濾

**目的**：過濾因隨機變異而表現好的不穩定因子

**參數**（`constants.py`）：
```python
BOOTSTRAP_N_ITERATIONS = 30      # 迭代次數
BOOTSTRAP_STABILITY_THRESHOLD = 0.75  # 穩定性閾值
BOOTSTRAP_SAMPLE_RATIO = 0.8     # 每次抽樣比例
```

**演算法**：
1. 重複 30 次：
   - 隨機抽樣 80% 資料
   - 計算每個因子的單因子 IC（Spearman）
   - 如果 |IC| > 0.02，該因子計數 +1
2. 只保留在 ≥75% 迭代中達標的因子

### CPCV (Combinatorial Purged Cross-Validation)

**目的**：透過多路徑驗證避免單一驗證期過擬合

**參數**（`constants.py`）：
```python
CPCV_N_FOLDS = 6              # 分割數
CPCV_N_TEST_FOLDS = 2         # 測試 fold 數 → C(6,2)=15 條路徑
CPCV_PURGE_DAYS = 5           # Purging 天數
CPCV_EMBARGO_DAYS = 5         # Embargo 天數
CPCV_CVPFI_THRESHOLD = 0.95   # P(importance > 0) 門檻
CPCV_TIME_DECAY_RATE = 0.95   # 時間衰減率
```

**CVPFI 方法**（`cpcv.py`）：

使用 Cross-Validated Permutation Feature Importance (CVPFI) 方法選擇因子：

```python
from scipy import stats

def calculate_cvpfi_probability(mean_importance, std_importance):
    """
    計算 CVPFI 概率

    假設 importance ~ N(μ, σ)，計算 P(importance > 0)。
    同時考慮效果大小（mean）和穩定性（std）。
    """
    if std_importance <= 0:
        return 1.0 if mean_importance > 0 else 0.0
    z_score = mean_importance / std_importance
    return stats.norm.cdf(z_score)
```

**範例**：

| mean | std | P(imp > 0) | 結果 |
|------|-----|------------|------|
| 0.02 | 0.01 | 0.977 | ✓ 通過 |
| 0.015 | 0.005 | 0.999 | ✓ 通過 |
| 0.01 | 0.02 | 0.691 | ✗ 不通過 |
| 0.005 | 0.02 | 0.599 | ✗ 不通過 |

**為什麼用 CVPFI 而非 t ≥ 3.0？**
- Harvey et al. (2016) 的 t ≥ 3.0 是用於**學術發表**的假發現率控制
- López de Prado 的 MDI/MDA 方法推薦 `importance > 0`
- CVPFI 來自 ACS Omega (2023) 論文，同時考慮效果大小和穩定性
- 更適合實際因子選擇場景

**選擇邏輯**：
1. **主選擇**：P(importance > 0) ≥ 0.95 + positive_ratio ≥ 0.6
2. **Fallback**（如果主選擇 < 5 個因子）：
   - P(importance > 0) ≥ 0.90 + positive_ratio ≥ 0.5
   - 補充到至少 5 個，最多 20 個因子

**時間衰減權重**：
- 近期路徑權重更高：`weight = 0.95^i`
- 有效樣本量使用 Kish's formula

---

## 模型訓練流程

### 訓練期配置

```python
TRAIN_DAYS = 252   # 訓練期：1 年交易日
VALID_DAYS = 5     # 驗證期：5 個交易日
EMBARGO_DAYS = 7   # Embargo：7 天（防止 label 洩漏）
```

### 資料處理流程

```python
# 1. 載入資料（擴展 end_date 確保 label 完整）
df = D.features(instruments, fields, start_time, end_time + 7days)

# 2. 特徵處理
X = process_inf(X)           # 處理無窮大值
X = zscore_by_date(X)        # 截面標準化
X = fillna(X, 0)             # 填補 NaN

# 3. Label 處理
y = rank_by_date(y)          # CSRankNorm（排名百分位）

# 4. 分割訓練/驗證集
X_train, X_valid = split_by_date(X, train_end, valid_start)
y_train, y_valid = split_by_date(y, train_end, valid_start)
```

### LightGBM 模型

**超參數來源**：
1. 從資料庫載入已培養的超參數
2. 如果沒有，使用保守預設值

**預設參數**：
```python
{
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "device": "gpu",
}
```

**訓練**：
```python
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)
```

### 模型輸出

儲存於 `data/models/{model_name}/`：
- `model.pkl`：訓練好的 LightGBM 模型
- `factors.json`：選出的因子清單
- `config.json`：訓練配置（日期、參數、IC 等）

---

## 回測流程

### Walk-Forward 回測

```
週 N 模型 ──→ 預測週 N+1 ──→ 計算 Live IC
週 N+1 模型 ──→ 預測週 N+2 ──→ 計算 Live IC
...
```

### 預測流程

```python
# 1. 載入模型和因子
model, factors, config = load_model(model_name)

# 2. 載入預測期特徵
X = D.features(instruments, factor_expressions, predict_start, predict_end)

# 3. 特徵處理（與訓練時相同）
X = process_inf(X)
X = zscore_by_date(X)
X = fillna(X, 0)

# 4. 預測
scores = model.predict(X)
```

### Live IC 計算

```python
# 計算 Label 對齊的收益率（T+1 → T+2）
returns = close.shift(-2) / close.shift(-1) - 1

# 計算 Spearman 相關係數（排名相關）
for date in common_dates:
    ic, _ = stats.spearmanr(scores[date], returns[date])
    daily_ics.append(ic)

live_ic = mean(daily_ics)
```

**關鍵**：收益率計算與 Label 定義對齊（T+1 → T+2）

### IC Decay 計算

```python
ic_decay = (valid_ic - live_ic) / valid_ic * 100
```

**預期範圍**：-30% ~ +30%（超過此範圍可能有問題）

### Top-K 選股

```python
# 使用第一天的預測分數
scores = predictions.loc[first_date]

# 排序選股（分數降序、代碼升序 tie-breaking）
topk_stocks = scores.sort_values(
    by=["score", "symbol"],
    ascending=[False, True]
).head(max_positions)
```

---

## IC 計算方法

### 統一使用 Spearman

| 位置 | 方法 | 說明 |
|------|------|------|
| 訓練驗證 IC | `corr(method="spearman")` | model_trainer.py |
| 單因子 IC | `corr(method="spearman")` | model_trainer.py |
| CPCV IC | `stats.spearmanr()` | cpcv.py |
| Live IC | `stats.spearmanr()` | walk_forward_backtester.py |
| Bootstrap IC | `corr(method="spearman")` | bootstrap_filter.py |

### 為什麼用 Spearman？

1. **與排名預測一致**：CSRankNorm 轉換成排名，Spearman 計算排名相關
2. **更穩健**：對離群值不敏感
3. **業界標準**：大多數量化研究使用 Rank IC

### IC 品質監控

```python
class TrainingQualityMetrics:
    factor_jaccard_sim: float  # 與上週的因子重疊度
    ic_moving_avg_5w: float    # 5 週 IC 移動平均
    ic_moving_std_5w: float    # 5 週 IC 標準差
    icir_5w: float             # 5 週 ICIR
```

**警報閾值**（`constants.py`）：
```python
QUALITY_JACCARD_MIN = 0.3  # Jaccard < 0.3 警報
QUALITY_IC_STD_MAX = 0.1   # IC 標準差 > 0.1 警報
QUALITY_ICIR_MIN = 0.5     # ICIR < 0.5 警報
```

---

## 關鍵參數配置

### constants.py 完整參數

```python
# === 時區 ===
TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# === 訓練期設定 ===
TRAIN_DAYS = 252      # 訓練期：1 年
VALID_DAYS = 5        # 驗證期：5 個交易日
EMBARGO_DAYS = 7      # Embargo：7 天

# === CPCV 參數 ===
CPCV_N_FOLDS = 6
CPCV_N_TEST_FOLDS = 2
CPCV_PURGE_DAYS = 5
CPCV_EMBARGO_DAYS = 5
CPCV_CVPFI_THRESHOLD = 0.95        # P(importance > 0) 門檻
CPCV_CVPFI_FALLBACK_THRESHOLD = 0.90  # Fallback 門檻
CPCV_TIME_DECAY_RATE = 0.95

# === CPCV 選擇參數 ===
CPCV_MIN_POSITIVE_RATIO = 0.6      # 主選擇穩定性門檻
CPCV_FALLBACK_POSITIVE_RATIO = 0.5   # Fallback 穩定性門檻
CPCV_FALLBACK_MAX_FACTORS = 20
CPCV_MIN_FACTORS_BEFORE_FALLBACK = 5

# === Bootstrap 穩定性 ===
BOOTSTRAP_N_ITERATIONS = 30
BOOTSTRAP_STABILITY_THRESHOLD = 0.75
BOOTSTRAP_SAMPLE_RATIO = 0.8

# === 品質監控 ===
QUALITY_JACCARD_MIN = 0.3
QUALITY_IC_STD_MAX = 0.1
QUALITY_ICIR_MIN = 0.5
```

---

## 參考文獻

### 學術論文

1. **Learning to Rank**
   - Poh, D., Lim, B., Zohren, S., & Roberts, S. (2021). "Building Cross-Sectional Systematic Strategies By Learning to Rank"
   - [arXiv:2012.07149](https://arxiv.org/abs/2012.07149)
   - 核心觀點：排名預測比收益預測更有效，Sharpe Ratio 提升 3 倍

2. **CVPFI (Cross-Validated Permutation Feature Importance)**
   - ACS Omega (2023). "Interpretation of Machine Learning Models for Data Sets with Many Features Using Feature Importance"
   - 核心觀點：計算 P(importance > 0)，同時考慮效果大小和穩定性
   - 論文使用 P > 0.997（三西格瑪），我們使用 P > 0.95

3. **CPCV**
   - López de Prado, M. (2018). "Advances in Financial Machine Learning"
   - 核心觀點：Combinatorial Purged Cross-Validation 避免過擬合

4. **Bootstrap Stability Selection**
   - Meinshausen, N. & Bühlmann, P. (2010). "Stability Selection"
   - 核心觀點：穩定性 ≥ 75% 的特徵才可靠

### Qlib 參考

- [Qlib Data Documentation](https://qlib.readthedocs.io/en/stable/component/data.html)
- [Qlib GRU Configuration](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml)
- [Qlib LightGBM Configuration](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)

---

## 更新歷史

| 日期 | 變更 |
|------|------|
| 2026-02-04 | CVPFI 方法：使用 P(importance > 0) 替代 t 統計量 |
| 2026-02-04 | 初版：統一 IC 計算（Spearman）、Label 使用 CSRankNorm |
