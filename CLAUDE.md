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

## 已完成

- [x] 資料同步（9 種資料集，TWSE/FinMind/yfinance）
- [x] Qlib 導出器（30 個欄位，PIT 月營收）
- [x] 因子管理（30 個預設因子，CRUD，驗證）
- [x] 前端（8 個頁面）
- [x] **模型訓練**（LightGBM，IC 增量選擇，按單因子 IC 排序）
- [x] **即時更新**（WebSocket + Zustand，CRUD 後自動刷新）
- [x] **超參數培養**（Walk Forward Optimization + Optuna）
- [x] **超參數管理 UI**（培養、選擇、刪除）
- [x] **因子數量自適應縮放**（sqrt(ratio) 縮放公式）
- [x] **回測系統**（backtrader 整合，Equity Curve，績效指標，K-line 圖表）

## 待完成

- [ ] **預測評分**（每日訊號產生）
- [ ] **排程系統**（每日自動同步+訓練）

## 關鍵規則

- 時間：`Asia/Taipei` (UTC+8)
- 股票池：市值前 100 大（排除 ETF、KY）
- 因子挑選：IC 增量選擇法
- **禁止自行啟動伺服器**

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

超參數培養時使用 30 個因子，但訓練時 IC 增量選擇只會選出 2-8 個因子。
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

## 資料來源

| 優先序 | 來源 | 限制 |
|--------|------|------|
| 1 | TWSE RWD | 當日 17:30 後 |
| 2 | FinMind | 600次/時 |
| 3 | yfinance | 無限制 |

**注意**：不要用 TWSE OpenAPI（`openapi.twse.com.tw`）

## 詳細文檔

- [API 設計](docs/api-design.md)
- [資料集](docs/datasets.md)
- [需求規格](docs/requirements.md)
