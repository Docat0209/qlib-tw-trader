# qlib-tw-trader

台灣股票交易與預測系統

## 專案目標

使用 qlib 進行台股預測，透過 IC 增量選擇法挑選因子，產生每日交易訊號。

## 快速指令

```bash
# 啟動後端
uvicorn src.interfaces.app:app --reload --port 8000

# 啟動前端
cd frontend && npm run dev

# Seed 因子
curl -X POST http://localhost:8000/api/v1/factors/seed

# 導出 qlib
curl -X POST http://localhost:8000/api/v1/qlib/export/sync \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2022-01-01","end_date":"2025-01-01"}'
```

## 關鍵規則

- 時間：`Asia/Taipei` (UTC+8)
- 股票池：市值前 100 大（排除 ETF、KY）
- 因子挑選：IC 增量選擇法
- **禁止自行啟動伺服器**
- **資料庫路徑**：`data/data.db`（不是 `data/trader.db`）

## DVC 資料版本控制

使用 DVC 管理大型檔案，透過 Google Drive Desktop 自動同步雲端。

### 追蹤的檔案

| 檔案 | 大小 | 說明 |
|------|------|------|
| `data/data.db` | ~131MB | SQLite 資料庫 |
| `data/models/` | ~19MB | 訓練好的模型（156 週） |

### 常用指令

```bash
# 上傳變更（訓練新模型或更新資料庫後）
python -m dvc push

# 下載資料（在新電腦或 clone 後）
python -m dvc pull

# 檢查狀態
python -m dvc status
```

### 雲端位置

- 本地：`G:\My Drive\qlib-tw-trader`
- Google Drive Desktop 自動同步至雲端

### 新電腦設定

1. 安裝 Google Drive Desktop 並登入
2. 確認同步資料夾掛載為 `G:\My Drive`
3. Clone repo 後執行 `python -m dvc pull`

## Qlib 資料架構

**重要**：Qlib `.bin` 檔案是從資料庫動態導出的，不是靜態資料。

### 資料流程

```
資料庫 (stock_daily, etc.)
    ↓
QlibExporter.export()  ← 指定日期範圍
    ↓
data/qlib/*.bin
    ↓
ModelTrainer / Backtester 使用
```

### 日期範圍判斷

- **正確**：查詢資料庫 `stock_daily` 表的 `MIN(date)` / `MAX(date)`
- **錯誤**：讀取現有 qlib 檔案的日期範圍（可能是舊的導出）

### 訓練/回測前

訓練和回測前，系統會自動調用 `QlibExporter` 導出所需日期範圍的資料：

```python
# 範例：訓練時自動導出
exporter = QlibExporter(session)
exporter.export(ExportConfig(
    start_date=train_start - lookback_days,  # 預留因子計算緩衝
    end_date=valid_end,
))
```

### 模型命名

格式：`YYYYMM-{hash}`
- `YYYYMM`：valid_end 的年月
- `hash`：6 位 MD5（基於 run_id + valid_end + factor_count）

例：`202502-a1b2c3`

## 超參數縮放原理

### 核心洞察

超參數培養時使用全部因子（目前 263 個），但訓練時 IC 增量選擇只會選出部分因子。
**超參數需要根據實際因子數量動態調整。**

### 為什麼可以縮放？

超參數控制的是「模型容量 vs 資料特性」的平衡，而非特定因子組合：

```
我們訓練的是：「我們的股票資料 + N 維輸入的最佳模型複雜度」
```

當因子從 30 減到 5 時：
- **不變**：樣本數量、噪音水平、標籤分佈
- **減少**：輸入維度 → 需要的模型容量

### 縮放公式

```python
# src/services/model_trainer.py
sqrt_ratio = sqrt(actual_factor_count / base_factor_count)

num_leaves: 31 → 8      # 模型容量隨維度縮小
max_depth: 5 → 3        # 交互深度減少
min_data_in_leaf: 44 → 11  # 允許更細的分割
lambda_l1/l2: 縮小      # 低維需較少正則化
```

### 效果

| 指標 | 縮放前 | 縮放後 |
|------|--------|--------|
| 選出因子數 | 2-4 | 5-8 |
| IC 範圍 | 0.03-0.04 | 0.05-0.06 |

**原因**：適當的模型複雜度讓系統能正確檢測因子貢獻，而非因過擬合提早停止選擇。

## 預測與交易時序

### Label 定義

```python
label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
```

- T 日特徵 → 預測 T+1 收盤 ~ T+2 收盤的收益率
- 若要捕捉此收益，應在 T+1 開盤買入

### 避免 Lookahead Bias

**Predictor:**
- `trade_date` = 預計交易日期（買入日）
- 系統自動使用 `trade_date - 1` 的特徵資料
- 返回 `feature_date`（實際使用的資料日期）

**Backtester:**
- 在 T 日交易時，使用 T-1 日的分數
- 預設使用 Open Price（更符合實際交易）

### 正確流程示例

要在 2/2 開盤買入：
1. 使用 2/1 收盤後的資料計算特徵
2. 模型預測 2/2→2/3 收益率
3. 2/2 開盤執行買入

## Top-K 選股與 Tie-Breaking

### 已知問題：模型區分能力不足

目前模型可能產生大量相同分數的股票（例如 100 支股票只有 7 種分數）。
這導致 Top-K 選股結果取決於資料順序，而非模型判斷。

### Tie-Breaking 機制

為確保結果穩定可重現，Predictor 和 Backtester 都採用相同的排序邏輯：

```python
# 先按分數降序，再按股票代碼升序
df.sort_values(by=["score", "symbol"], ascending=[False, True]).head(top_k)
```

### 因子擴充（已實施）

已將因子從 30 個擴充至 263 個，包含：
- Alpha158 純 K 線因子（109 個）
- 台股籌碼因子（107 個）
- 交互因子（50 個）

需要重新訓練模型以驗證區分能力是否提升。

## 資料來源

| 優先序 | 來源 | 限制 |
|--------|------|------|
| 1 | TWSE RWD | 當日 17:30 後 |
| 2 | FinMind | 600次/時 |
| 3 | yfinance | 無限制 |

**注意**：不要用 TWSE OpenAPI（`openapi.twse.com.tw`）
