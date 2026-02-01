# API 設計文件

## 設計原則

- RESTful 風格
- 回傳格式統一為 JSON
- 分頁使用 `?page=1&size=20`
- 錯誤回傳 `{ "detail": "錯誤訊息" }`
- 非同步任務進度透過 **WebSocket** 推送

## 設計決策

| 項目 | 決策 |
|------|------|
| 因子公式格式 | Python 表達式字串 |
| 任務進度通知 | WebSocket 推送 |
| 持倉管理 | 純展示（依賴模型預測）|
| 多模型支援 | 單一 active 模型 |

## 核心概念

### 因子 (Factor)
- 不記錄個體 IC（個體 IC 不代表模型 IC）
- 記錄 **入選率**：在歷次模型訓練中被選中的機率
- 用於手動剔除低價值因子

### 模型 (Model)
- 只保留 **最新模型檔案**
- 歷史模型只保留 **metadata**（用於比較）
- 關鍵指標：因子數、IC、ICIR、訓練/驗證時間

---

## 0. System API ✅

### GET /api/v1/system/health ✅

健康檢查。

### GET /api/v1/system/data-status ✅

資料完整度狀態。

---

## 1. Dashboard API

### GET /api/v1/dashboard/summary

首頁摘要資訊。

**Response:**
```json
{
  "factors": {
    "total": 15,
    "enabled": 12,
    "low_selection_count": 2
  },
  "model": {
    "last_trained_at": "2026-01-15T10:30:00",
    "days_since_training": 15,
    "needs_retrain": false,
    "factor_count": 12,
    "ic": 0.058,
    "icir": 1.45
  },
  "prediction": {
    "date": "2026-01-30",
    "buy_signals": 3,
    "sell_signals": 2,
    "top_pick": { "symbol": "2330", "score": 0.85 }
  },
  "data_status": {
    "is_complete": true,
    "last_updated": "2026-01-30T15:30:00",
    "missing_count": 0
  },
  "performance": {
    "today_return": 0.012,
    "mtd_return": 0.035,
    "ytd_return": 0.082,
    "total_return": 0.156
  }
}
```

**欄位說明:**
- `low_selection_count`: 入選率低於閾值（如 30%）的因子數量，提示可考慮剔除

---

## 2. Factors API ✅

### GET /api/v1/factors ✅

取得因子清單。

**Query Parameters:**
- `category` (optional): 篩選類別 (technical, fundamental, institutional)
- `enabled` (optional): true/false

**Response:**
```json
{
  "items": [
    {
      "id": "f001",
      "name": "momentum_20d",
      "display_name": "20日動量",
      "category": "technical",
      "description": "過去20日報酬率",
      "selection_rate": 0.85,
      "times_selected": 17,
      "times_evaluated": 20,
      "enabled": true,
      "created_at": "2025-06-01T00:00:00"
    }
  ],
  "total": 15
}
```

**欄位說明:**
- `selection_rate`: 入選率 = times_selected / times_evaluated
- `times_selected`: 被模型選中的次數
- `times_evaluated`: 參與訓練評估的總次數

### GET /api/v1/factors/{factor_id} ✅

取得單一因子詳情。

**Response:**
```json
{
  "id": "f001",
  "name": "momentum_20d",
  "display_name": "20日動量",
  "category": "technical",
  "description": "過去20日報酬率",
  "formula": "(close - close.shift(20)) / close.shift(20)",
  "selection_rate": 0.85,
  "times_selected": 17,
  "times_evaluated": 20,
  "enabled": true,
  "created_at": "2025-06-01T00:00:00",
  "selection_history": [
    { "model_id": "m001", "trained_at": "2026-01-15", "selected": true },
    { "model_id": "m002", "trained_at": "2025-12-15", "selected": true },
    { "model_id": "m003", "trained_at": "2025-11-15", "selected": false }
  ]
}
```

### POST /api/v1/factors ✅

新增因子。

**Request:**
```json
{
  "name": "rsi_14d",
  "display_name": "14日RSI",
  "category": "technical",
  "description": "14日相對強弱指標",
  "formula": "ta.RSI(close, 14)"
}
```

### PUT /api/v1/factors/{factor_id} ✅

更新因子。

### DELETE /api/v1/factors/{factor_id} ✅

刪除因子。

### PATCH /api/v1/factors/{factor_id}/toggle ✅

切換因子啟用狀態。

### POST /api/v1/factors/validate ✅

驗證因子表達式語法。

### POST /api/v1/factors/seed ✅

插入 30 個預設因子。

### GET /api/v1/factors/available ✅

取得可用欄位和運算符。

---

## 3. Models API

### GET /api/v1/models/current

取得當前 active 模型。

**Response:**
```json
{
  "id": "m020",
  "trained_at": "2026-01-15T10:30:00",
  "factor_count": 12,
  "factors": ["momentum_20d", "rsi_14d", "volume_ratio", "..."],
  "train_period": { "start": "2020-01-01", "end": "2025-10-31" },
  "valid_period": { "start": "2025-11-01", "end": "2025-12-31" },
  "metrics": {
    "ic": 0.058,
    "icir": 1.45
  },
  "training_duration_seconds": 3600
}
```

### GET /api/v1/models/history

取得歷史模型 metadata（用於比較）。

**Query Parameters:**
- `limit` (optional): 筆數，預設 20

**Response:**
```json
{
  "items": [
    {
      "id": "m020",
      "trained_at": "2026-01-15T10:30:00",
      "factor_count": 12,
      "train_period": { "start": "2020-01-01", "end": "2025-10-31" },
      "valid_period": { "start": "2025-11-01", "end": "2025-12-31" },
      "metrics": { "ic": 0.058, "icir": 1.45 },
      "is_current": true
    },
    {
      "id": "m019",
      "trained_at": "2025-12-15T09:00:00",
      "factor_count": 11,
      "train_period": { "start": "2020-01-01", "end": "2025-09-30" },
      "valid_period": { "start": "2025-10-01", "end": "2025-11-30" },
      "metrics": { "ic": 0.052, "icir": 1.38 },
      "is_current": false
    }
  ],
  "total": 20
}
```

### GET /api/v1/models/comparison

取得模型指標比較（用於圖表）。

**Response:**
```json
{
  "models": [
    { "id": "m020", "trained_at": "2026-01-15", "ic": 0.058, "icir": 1.45, "factor_count": 12 },
    { "id": "m019", "trained_at": "2025-12-15", "ic": 0.052, "icir": 1.38, "factor_count": 11 },
    { "id": "m018", "trained_at": "2025-11-15", "ic": 0.055, "icir": 1.42, "factor_count": 10 }
  ]
}
```

### GET /api/v1/models/status

取得訓練狀態（用於檢查是否需要重訓）。

**Response:**
```json
{
  "last_trained_at": "2026-01-15T10:30:00",
  "days_since_training": 15,
  "needs_retrain": false,
  "retrain_threshold_days": 30,
  "current_job": null
}
```

### POST /api/v1/models/train

觸發模型訓練（非同步，透過 WebSocket 推送進度）。

**Request:**
```json
{
  "train_end": "2025-10-31",
  "valid_end": "2025-12-31"
}
```

**Response:**
```json
{
  "job_id": "train_abc123",
  "status": "queued",
  "message": "訓練任務已排入佇列"
}
```

---

## 3.5 Hyperparams API ✅

超參數培養與管理。

### GET /api/v1/hyperparams ✅

取得超參數組列表。

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "name": "hp-2025-01",
      "cultivated_at": "2025-01-20T10:00:00",
      "n_periods": 5,
      "is_current": true,
      "learning_rate": 0.01,
      "num_leaves": 31
    }
  ],
  "total": 3
}
```

### GET /api/v1/hyperparams/current ✅

取得當前使用的超參數組。

### GET /api/v1/hyperparams/{hp_id} ✅

取得超參數組詳情（含穩定性指標、各窗口結果）。

**Response:**
```json
{
  "id": 1,
  "name": "hp-2025-01",
  "cultivated_at": "2025-01-20T10:00:00",
  "n_periods": 5,
  "is_current": true,
  "params": {
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.01,
    "min_data_in_leaf": 44,
    "lambda_l1": 0.52,
    "lambda_l2": 4.16
  },
  "stability": {
    "num_leaves": 0.15,
    "learning_rate": 0.87,
    "lambda_l1": 1.55
  },
  "periods": [
    {
      "train_start": "2022-01-01",
      "train_end": "2024-01-01",
      "valid_start": "2024-01-02",
      "valid_end": "2024-03-31",
      "best_ic": 0.052
    }
  ]
}
```

**穩定性 (stability) 說明：**
- CV < 0.3：穩定（綠）
- CV 0.3-0.4：中等（橙）
- CV > 0.4：不穩定（紅）

### POST /api/v1/hyperparams ✅

執行超參數培養（非同步，Walk Forward Optimization + Optuna）。

**Request:**
```json
{
  "name": "hp-2025-02",
  "n_periods": 5,
  "n_trials_per_period": 20
}
```

**欄位說明：**
- `n_periods`: Walk Forward 窗口數（3-10）
- `n_trials_per_period`: 每窗口 Optuna trials（10-50）

**Response:**
```json
{
  "job_id": "cultivate_abc123",
  "status": "queued",
  "message": "超參數培養任務已排入佇列"
}
```

*進度透過 WebSocket 推送*

### PATCH /api/v1/hyperparams/{hp_id}/current ✅

設為當前使用的超參數組（會清除其他 is_current）。

### PATCH /api/v1/hyperparams/{hp_id} ✅

更新超參數組名稱。

### DELETE /api/v1/hyperparams/{hp_id} ✅

刪除超參數組（不能刪除當前使用的）。

---

## 4. Backtest API

### POST /api/v1/backtest/run

執行回測（非同步）。

**Request:**
```json
{
  "model_id": "m001",
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "initial_capital": 1000000,
  "max_positions": 10
}
```

**Response:**
```json
{
  "job_id": "bt_abc123",
  "status": "queued"
}
```

### GET /api/v1/backtest/{backtest_id}

取得回測結果（任務完成後）。

**Response:**
```json
{
  "id": "bt001",
  "model_id": "m020",
  "period": { "start": "2025-01-01", "end": "2025-12-31" },
  "initial_capital": 1000000,
  "result": {
    "total_return": 0.235,
    "annualized_return": 0.235,
    "sharpe_ratio": 1.65,
    "max_drawdown": -0.089,
    "win_rate": 0.58,
    "trades_count": 156
  },
  "equity_curve": [
    { "date": "2025-01-02", "value": 1000000 },
    { "date": "2025-01-03", "value": 1012000 }
  ],
  "created_at": "2026-01-30T15:00:00"
}
```

*進度透過 WebSocket 推送*

### GET /api/v1/backtest/history

取得歷史回測紀錄。

**Response:**
```json
{
  "items": [
    {
      "id": "bt001",
      "model_name": "lightgbm_v1",
      "period": { "start": "2025-01-01", "end": "2025-12-31" },
      "total_return": 0.235,
      "sharpe_ratio": 1.65,
      "created_at": "2026-01-20T09:00:00"
    }
  ],
  "total": 5
}
```

---

## 5. Positions / Predictions API

### GET /api/v1/positions

取得當前持倉。

**Response:**
```json
{
  "as_of": "2026-01-30",
  "total_value": 1250000,
  "cash": 150000,
  "positions": [
    {
      "symbol": "2330",
      "name": "台積電",
      "shares": 1000,
      "avg_cost": 580.5,
      "current_price": 625.0,
      "market_value": 625000,
      "unrealized_pnl": 44500,
      "unrealized_pnl_pct": 0.0767,
      "weight": 0.50
    }
  ]
}
```

### GET /api/v1/predictions/latest

取得最新預測信號。

**Response:**
```json
{
  "date": "2026-01-30",
  "model_id": "m001",
  "signals": [
    {
      "symbol": "2330",
      "name": "台積電",
      "score": 0.85,
      "rank": 1,
      "signal": "buy",
      "current_position": 1000
    },
    {
      "symbol": "2317",
      "name": "鴻海",
      "score": -0.65,
      "rank": 50,
      "signal": "sell",
      "current_position": 500
    }
  ]
}
```

### GET /api/v1/predictions/history

取得歷史預測紀錄。

**Query Parameters:**
- `start_date`, `end_date`
- `symbol` (optional)

### GET /api/v1/trades

取得交易紀錄。

**Response:**
```json
{
  "items": [
    {
      "id": "t001",
      "date": "2026-01-29",
      "symbol": "2330",
      "name": "台積電",
      "side": "buy",
      "shares": 1000,
      "price": 580.5,
      "amount": 580500,
      "commission": 825,
      "reason": "模型預測買入信號"
    }
  ],
  "total": 156
}
```

---

## 6. Data Status API ✅

### GET /api/v1/data/status ✅

取得資料完整度狀態。

**Response:**
```json
{
  "last_checked": "2026-01-30T15:30:00",
  "overall_complete": true,
  "tables": [
    {
      "name": "daily_price",
      "display_name": "日K線",
      "last_date": "2026-01-30",
      "expected_date": "2026-01-30",
      "is_complete": true,
      "record_count": 8880,
      "update_time": "~15:00"
    },
    {
      "name": "institutional",
      "display_name": "三大法人",
      "last_date": "2026-01-29",
      "expected_date": "2026-01-30",
      "is_complete": false,
      "record_count": 7400,
      "update_time": "~16:30"
    }
  ],
  "stocks_tracked": ["2330"]
}
```

### POST /api/v1/data/sync ✅

手動觸發資料同步。

**Request:**
```json
{
  "tables": ["daily_price", "institutional"],
  "symbols": ["2330"]
}
```

**Response:**
```json
{
  "job_id": "sync_abc123",
  "status": "queued"
}
```

*同步進度透過 WebSocket 推送*

### GET /api/v1/data/sync/history

取得資料同步歷史。

**Response:**
```json
{
  "items": [
    {
      "id": "sync001",
      "started_at": "2026-01-30T15:00:00",
      "completed_at": "2026-01-30T15:05:00",
      "tables_synced": ["daily_price", "institutional"],
      "records_added": 24,
      "status": "completed"
    }
  ],
  "total": 10
}
```

---

## 6.5 Qlib Export API ✅

### POST /api/v1/qlib/export/sync ✅

同步導出 qlib .bin 格式資料。

**Request:**
```json
{
  "start_date": "2022-01-01",
  "end_date": "2025-01-01"
}
```

### GET /api/v1/qlib/export/status ✅

取得導出狀態。

### GET /api/v1/qlib/export/fields ✅

取得可用欄位列表（30 個欄位）。

---

## 7. Performance API

### GET /api/v1/performance/summary

取得收益摘要。

**Response:**
```json
{
  "as_of": "2026-01-30",
  "returns": {
    "today": 0.012,
    "wtd": 0.025,
    "mtd": 0.035,
    "ytd": 0.082,
    "total": 0.156
  },
  "benchmark_returns": {
    "today": 0.008,
    "wtd": 0.015,
    "mtd": 0.022,
    "ytd": 0.045
  },
  "alpha": {
    "mtd": 0.013,
    "ytd": 0.037
  }
}
```

### GET /api/v1/performance/equity-curve

取得權益曲線。

**Query Parameters:**
- `start_date`, `end_date`
- `benchmark` (optional): true/false 是否包含大盤

**Response:**
```json
{
  "data": [
    {
      "date": "2025-01-02",
      "portfolio_value": 1000000,
      "cumulative_return": 0,
      "benchmark_return": 0
    },
    {
      "date": "2025-01-03",
      "portfolio_value": 1012000,
      "cumulative_return": 0.012,
      "benchmark_return": 0.008
    }
  ]
}
```

### GET /api/v1/performance/monthly

取得月報酬。

**Response:**
```json
{
  "data": [
    { "year": 2025, "month": 1, "return": 0.035, "benchmark": 0.022 },
    { "year": 2025, "month": 2, "return": -0.012, "benchmark": -0.018 }
  ]
}
```

---

## 8. WebSocket API

訓練、回測、同步等耗時任務透過 WebSocket 推送進度。

### 連線

```
ws://localhost:8000/api/v1/ws
```

### 訊息格式

**任務進度更新:**
```json
{
  "type": "job_progress",
  "job_id": "train_abc123",
  "job_type": "train",
  "status": "running",
  "progress": 45,
  "message": "評估因子 IC (8/12)",
  "started_at": "2026-01-30T14:00:00"
}
```

**任務完成:**
```json
{
  "type": "job_completed",
  "job_id": "train_abc123",
  "job_type": "train",
  "status": "completed",
  "result": {
    "model_id": "m021",
    "factor_count": 13,
    "ic": 0.061,
    "icir": 1.52
  },
  "completed_at": "2026-01-30T15:00:00"
}
```

**任務失敗:**
```json
{
  "type": "job_failed",
  "job_id": "train_abc123",
  "job_type": "train",
  "status": "failed",
  "error": "記憶體不足",
  "failed_at": "2026-01-30T14:30:00"
}
```

### 任務類型 (job_type)

| 類型 | 說明 |
|------|------|
| `train` | 模型訓練 |
| `backtest` | 回測 |
| `sync` | 資料同步 |

### 非同步任務流程

1. POST 請求建立任務，回傳 `job_id`
2. WebSocket 推送進度更新
3. 任務狀態：`queued` → `running` → `completed` / `failed`
