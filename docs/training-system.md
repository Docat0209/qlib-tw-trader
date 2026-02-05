# 訓練與回測系統技術文檔

本文檔詳細說明系統的模型訓練流程、回測流程、使用的方法、模型和參數。

## 目錄

1. [系統架構概覽](#系統架構概覽)
2. [API 端點](#api-端點)
3. [Label 定義與處理](#label-定義與處理)
4. [因子選擇流程](#因子選擇流程)
5. [模型訓練流程](#模型訓練流程)
6. [增量學習](#增量學習)
7. [回測流程](#回測流程)
8. [IC 計算方法](#ic-計算方法)
9. [超參數培養](#超參數培養)
10. [關鍵參數配置](#關鍵參數配置)
11. [參考文獻](#參考文獻)

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
│  │              因子選擇（三種模式）                          │   │
│  │  "dedup": RD-Agent IC 去重複（預設）                      │   │
│  │  "none": Qlib 標準（不選擇）                              │   │
│  │  "cpcv": CPCV + permutation importance（備用）            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                    │                            │
│                                    ▼                            │
│                           LightGBM 訓練                         │
│                                    │                            │
│                                    ▼                            │
│                           驗證期 IC 計算                         │
│                                    │                            │
│                                    ▼                            │
│                    增量學習（驗證期微調）                         │
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

## API 端點

### 模型訓練 API

| 端點 | 方法 | 功能 |
|------|------|------|
| `/api/v1/models/train` | POST | 觸發單週訓練（week_id） |
| `/api/v1/models/train-batch` | POST | 批量訓練整年模型 |
| `/api/v1/models/weeks` | GET | 取得可訓練週狀態 |
| `/api/v1/models/status` | GET | 取得訓練狀態 |
| `/api/v1/models/data-range` | GET | 取得資料庫日期範圍 |

### 超參數管理 API

| 端點 | 方法 | 功能 |
|------|------|------|
| `/api/v1/hyperparams` | POST | 建立並培養超參數 |
| `/api/v1/hyperparams` | GET | 列出所有超參數組 |
| `/api/v1/hyperparams/{hp_id}` | GET | 取得超參數詳情 |
| `/api/v1/hyperparams/{hp_id}` | DELETE | 刪除超參數組 |

### 回測 API

| 端點 | 方法 | 功能 |
|------|------|------|
| `/api/v1/backtests/walk-forward` | POST | Walk-Forward 回測 |
| `/api/v1/backtests` | GET | 列出回測記錄 |

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

### Qlib 官方配置差異

| 模型 | Qlib 官方 Label 標準化 | 我們的配置 |
|------|----------------------|-----------|
| **LightGBM** | CSZScoreNorm | CSRankNorm |
| **LSTM/GRU** | CSRankNorm | - |

**為什麼我們選擇 CSRankNorm（而非官方 LightGBM 的 CSZScoreNorm）？**

1. **與評估指標一致**：我們使用 Spearman IC（排名相關），CSRankNorm 讓模型學習預測排名
2. **更穩健**：CSRankNorm 對離群值不敏感，台股小型股常有極端收益
3. **訓練-評估對齊**：模型學習排名 → IC 評估排名相關 → 選股基於排名

**重要**：無論選擇哪種方法，**訓練、增量更新、預測必須使用相同的標準化**：

| 階段 | 特徵標準化 | Label 標準化 |
|------|-----------|-------------|
| 訓練 | CSZScoreNorm | CSRankNorm |
| 增量更新 | CSZScoreNorm | CSRankNorm |
| 預測 | CSZScoreNorm | - |
| Live IC 計算 | - | 原始收益（對 score 排名計算 Spearman）|

---

## 因子選擇流程

### 三種選擇模式

系統支援三種文獻支持的因子選擇方法：

| 模式 | 來源 | 說明 | 預設 |
|------|------|------|------|
| `dedup` | RD-Agent (Microsoft Research, 2025) | IC 去重複，移除高相關因子 | ✓ |
| `none` | Qlib (Microsoft) | 不做選擇，依賴 LightGBM 內建機制 | |
| `cpcv` | López de Prado (2018) | CPCV + permutation importance | 備用 |

### RD-Agent IC 去重複（預設方法）

**來源**：[R&D-Agent-Quant](https://arxiv.org/html/2505.15155v2) (Microsoft Research, 2025)

**原理**：
> "New factors with IC_max(n) ≥ 0.99 are deemed redundant and excluded."
> 結果：減少 70% 因子，ARR 提升 2 倍

**算法**：

```
266 個候選因子
       │
       ▼
┌─────────────────────────────┐
│ 1. 計算因子間相關係數矩陣     │
│    corr_matrix = X.corr()   │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ 2. 計算每個因子對 label 的 IC │
│    single_ic = X[f].corr(y) │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ 3. 按 IC 降序排列            │
│    優先保留高 IC 因子         │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ 4. 貪婪去重複                │
│    若 corr >= 0.99 則移除    │
└─────────────────────────────┘
       │
       ▼
   ~150-200 個非冗餘因子
```

**實作**（`src/services/factor_selection/ic_dedup.py`）：

```python
class ICDeduplicator:
    def __init__(self, correlation_threshold: float = 0.99):
        self.threshold = correlation_threshold

    def deduplicate(self, factors, X, y):
        # 1. 計算相關矩陣
        corr_matrix = X[factor_names].corr(method="spearman")

        # 2. 計算單因子 IC
        single_ics = {f.name: X[f.name].corr(y, method="spearman") for f in factors}

        # 3. 按 IC 降序排列
        sorted_factors = sorted(factors, key=lambda f: abs(single_ics[f.name]), reverse=True)

        # 4. 貪婪去重複
        kept = []
        for factor in sorted_factors:
            is_redundant = False
            for kept_factor in kept:
                if abs(corr_matrix.loc[factor.name, kept_factor.name]) >= self.threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept.append(factor)

        return kept, stats
```

**使用**：

```python
from src.services.factor_selection import RobustFactorSelector

# 預設使用 IC 去重複
selector = RobustFactorSelector()  # method="dedup"
result = selector.select(factors, X, y)

# 或明確指定
selector = RobustFactorSelector(method="dedup", dedup_threshold=0.99)
```

### Qlib 標準（無選擇）

**來源**：[Microsoft Qlib](https://github.com/microsoft/qlib)

**原理**：
- 使用全部因子，依賴 LightGBM 內建機制
- LightGBM 會自動處理無用因子（feature_fraction、正則化）

```python
selector = RobustFactorSelector(method="none")
```

### CPCV（備用）

**來源**：López de Prado, M. (2018). "Advances in Financial Machine Learning"

**參數**：
```python
CPCV_N_FOLDS = 6              # 分割數
CPCV_N_TEST_FOLDS = 2         # 測試 fold 數 → C(6,2)=15 條路徑
CPCV_PURGE_DAYS = 5           # Purging 天數
CPCV_EMBARGO_DAYS = 5         # Embargo 天數
CPCV_MIN_POSITIVE_RATIO = 0.5 # 最低正向路徑比例
```

```python
selector = RobustFactorSelector(method="cpcv")
```

---

## 模型訓練流程

### 完整訓練步驟

```
POST /api/v1/models/train { "week_id": "2026W05" }
                │
                ▼
        ┌───────────────┐
        │ 1. 導出 Qlib  │  export_start = train_start - 90 days
        │    資料       │  （因子計算緩衝）
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 2. 載入資料   │  D.features() 讀取
        │              │  Label: Ref($close,-2)/Ref($close,-1)-1
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 3. 資料處理   │  process_inf → zscore_by_date → fillna
        │              │  Label: rank_by_date (CSRankNorm)
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 4. 因子選擇   │  RobustFactorSelector(method="dedup")
        │              │  IC 去重複：266 → ~150-200 因子
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 5. LightGBM   │  使用培養或預設超參數
        │    訓練       │  early_stopping=50
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 6. 計算       │  Spearman IC（驗證期）
        │    Valid IC   │  報告此 IC
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 7. 增量更新   │  使用驗證期資料微調
        │              │  num_boost_round=50
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ 8. 保存模型   │  model.pkl, factors.json, config.json
        └───────────────┘
```

### 訓練期配置

```python
TRAIN_DAYS = 252   # 訓練期：1 年交易日
VALID_DAYS = 5     # 驗證期：5 個交易日
EMBARGO_DAYS = 7   # Embargo：7 天（防止 label 洩漏）
```

### 資料處理

```python
# 1. 載入資料（擴展 end_date 確保 label 完整）
df = D.features(instruments, fields, start_time, end_time + 7days)

# 2. 特徵處理
X = process_inf(X)           # 處理無窮大值（用均值替換）
X = zscore_by_date(X)        # 截面標準化
X = fillna(X, 0)             # 填補 NaN

# 3. Label 處理
y = rank_by_date(y)          # CSRankNorm（排名百分位 [0, 1]）

# 4. 分割訓練/驗證集（按日期）
train_mask = (dates >= train_start) & (dates <= train_end)
valid_mask = (dates >= valid_start) & (dates <= valid_end)
```

### LightGBM 預設參數

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

### 模型輸出

儲存於 `data/models/{model_name}/`：

| 檔案 | 內容 |
|------|------|
| `model.pkl` | LightGBM Booster（增量更新後） |
| `factors.json` | 選出的因子清單 |
| `config.json` | 訓練配置、日期、IC、超參數 |

---

## 增量學習

### 設計原則

| 層級 | 決定什麼 | 更新頻率 | 方法 |
|------|---------|---------|------|
| **因子選擇** | 用哪些因子 | 每週 | IC 去重複 |
| **模型權重** | 因子怎麼組合 | 每日/每週 | `init_model` 增量學習 |

**核心原理**：
- 因子結構變化慢（週/月級）→ 每週重訓選擇
- 因子權重變化快（日級）→ 增量微調

### 訓練後增量更新

訓練完成後，自動使用驗證期資料進行增量更新：

```python
# 使用驗證期資料微調
# 重要：Label 使用 CSRankNorm（與主訓練流程一致）
y_valid_rank = self._rank_by_date(y_valid_incr)
valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid_rank.values)

incremented_model = lgb.train(
    incr_params,
    valid_data,
    num_boost_round=50,      # 少量更新
    init_model=best_model,   # 從訓練的模型開始
    keep_training_booster=True,
)

# 保存增量更新後的模型 A'
# 但報告的 IC 是模型 A 的驗證期 IC（未增量前）
```

### Walk-Forward 回測中的增量學習

```python
POST /api/v1/backtests/walk-forward
{
    "enable_incremental": true,   # 啟用每日增量
    "start_week_id": "2026W01",
    "end_week_id": "2026W10"
}
```

詳見 [增量學習設計文檔](incremental-learning-design.md)。

---

## 回測流程

### Walk-Forward 回測

```
週 N 模型 ──→ 預測週 N+1 ──→ 計算 Live IC
週 N+1 模型 ──→ 預測週 N+2 ──→ 計算 Live IC
...
                │
                ▼
          聚合統計
     ├─ 平均 Valid IC
     ├─ 平均 Live IC
     ├─ IC Decay %
     └─ 績效指標
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

### IC Decay 計算

```python
ic_decay = (valid_ic - live_ic) / valid_ic * 100
```

**預期範圍**：-30% ~ +30%

### Top-K 選股

```python
# 排序選股（分數降序、代碼升序 tie-breaking）
topk_stocks = scores.sort_values(
    by=["score", "symbol"],
    ascending=[False, True]
).head(max_positions)
```

---

## IC 計算方法

### 統一使用 Spearman

| 位置 | 方法 | 檔案 |
|------|------|------|
| 訓練驗證 IC | `corr(method="spearman")` | model_trainer.py |
| 單因子 IC | `corr(method="spearman")` | ic_dedup.py |
| Live IC | `stats.spearmanr()` | walk_forward_backtester.py |

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

**警報閾值**：
```python
QUALITY_JACCARD_MIN = 0.3  # Jaccard < 0.3 警報
QUALITY_IC_STD_MAX = 0.1   # IC 標準差 > 0.1 警報
QUALITY_ICIR_MIN = 0.5     # ICIR < 0.5 警報
```

---

## 超參數培養

### Walk-Forward Optimization

**方法**：多窗口優化 + 中位數聚合

```
1. 生成 N 個歷史窗口（預設 5）
        │
        ▼
2. 每個窗口獨立 Optuna 優化
   ├─ n_trials_per_period: 20
   ├─ 搜索：num_leaves, max_depth, learning_rate 等
   └─ 計算該窗口最佳 IC
        │
        ▼
3. 聚合中位數
   ├─ 取各窗口最佳參數的中位數
   └─ 計算穩定性（變異係數 CV）
        │
        ▼
4. 保存到資料庫
```

**API**：

```python
POST /api/v1/hyperparams
{
    "name": "hp_2026_02_v1",
    "n_periods": 5,
    "n_trials_per_period": 20
}
```

---

## 關鍵參數配置

### constants.py

```python
# === 時區 ===
TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# === 訓練期設定 ===
TRAIN_DAYS = 252      # 訓練期：1 年
VALID_DAYS = 5        # 驗證期：5 個交易日
EMBARGO_DAYS = 7      # Embargo：7 天

# === IC Deduplication (RD-Agent) ===
IC_DEDUP_THRESHOLD = 0.99  # 去重複閾值

# === CPCV 參數（備用）===
CPCV_N_FOLDS = 6
CPCV_N_TEST_FOLDS = 2
CPCV_PURGE_DAYS = 5
CPCV_EMBARGO_DAYS = 5
CPCV_MIN_POSITIVE_RATIO = 0.5
CPCV_TIME_DECAY_RATE = 0.95

# === 品質監控 ===
QUALITY_JACCARD_MIN = 0.3
QUALITY_IC_STD_MAX = 0.1
QUALITY_ICIR_MIN = 0.5
```

---

## 參考文獻

### 因子選擇

1. **RD-Agent** (Microsoft Research, 2025)
   - [arXiv:2505.15155](https://arxiv.org/html/2505.15155v2)
   - IC 去重複：IC_max >= 0.99 視為冗餘
   - 減少 70% 因子，ARR 提升 2 倍

2. **Qlib** (Microsoft)
   - [GitHub](https://github.com/microsoft/qlib)
   - 標準流程：無因子選擇，依賴 LightGBM

3. **López de Prado** (2018)
   - "Advances in Financial Machine Learning"
   - CPCV、MDI/MDA 方法

### 排名預測

4. **Learning to Rank**
   - Poh et al. (2021). "Building Cross-Sectional Systematic Strategies By Learning to Rank"
   - [arXiv:2012.07149](https://arxiv.org/abs/2012.07149)
   - 排名預測比收益預測更有效，Sharpe Ratio 提升 3 倍

### Qlib 參考

- [Qlib Data Documentation](https://qlib.readthedocs.io/en/stable/component/data.html)
- [Qlib LightGBM Configuration](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)

---

## 已知問題與修復

### 2026-02-05 發現：標籤二次標準化導致 IC = 0

**症狀**：
- 模型 IC = 0（精確值）
- Feature Importance 全為 0
- LightGBM 樹只有 1 個葉節點（常數預測）

**根本原因**：

```
標籤處理流程（錯誤）：
1. _prepare_train_valid_data: y = _rank_by_date(y)  → 排名 [0, 1]
2. _robust_factor_selection: y = _zscore_by_date(y)  → 二次標準化 ❌

問題：對已排名的標籤再做 z-score，信號完全丟失
```

**修復**：

```python
# 修復前（錯誤）
y_train_zscore = self._zscore_by_date(y_train_clean.to_frame()).squeeze()

# 修復後（正確）
y_train_final = y_train_clean  # 直接使用排名後的標籤
```

**位置**：`src/services/model_trainer.py` 第 1084-1091 行

### 其他潛在問題（待驗證）

| 問題 | 影響 | 優先級 |
|------|------|--------|
| NaN 填補為 0 | 污染訓練數據 | P1 |
| 因子平均 IC 極低（~0.000024） | 基礎信號不足 | P1 |
| 超參數與因子數不匹配 | 模型性能 | P2 |
| qlib 日期延伸邏輯 | 標籤計算誤差 | P2 |

### 診斷工具

檢查模型是否正常：

```python
import pickle
from pathlib import Path

model_file = Path('data/models/{model_name}/model.pkl')
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# 檢查樹結構
info = model.dump_model()
print('num_trees:', info.get('num_trees'))
print('first tree leaves:', info['tree_info'][0].get('num_leaves'))

# 檢查特徵重要性
imp = model.feature_importance()
print('non-zero importances:', (imp > 0).sum(), '/', len(imp))
```

**正常模型應該**：
- `num_trees` > 1
- `first tree leaves` > 1
- `non-zero importances` > 0

---

## 更新歷史

| 日期 | 變更 |
|------|------|
| 2026-02-05 | 修復：移除標籤二次標準化（IC = 0 問題） |
| 2026-02-05 | 新增：已知問題與修復章節 |
| 2026-02-04 | 重構：使用 RD-Agent IC 去重複替代 Bootstrap + CPCV |
| 2026-02-04 | 新增：三種因子選擇模式（dedup、none、cpcv） |
| 2026-02-04 | 新增：增量學習章節 |
| 2026-02-04 | 新增：API 端點章節 |
| 2026-02-04 | 初版：統一 IC 計算（Spearman）、Label 使用 CSRankNorm |
