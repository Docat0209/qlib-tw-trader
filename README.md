# qlib-tw-trader

台灣股票預測系統，基於 Microsoft qlib + LightGBM。

## 功能特色

| 功能 | 說明 |
|------|------|
| **資料同步** | 9 種資料集，自動抓取 TWSE/FinMind/yfinance |
| **因子管理** | 263 個預設因子（Alpha158 + 籌碼 + 交互），支援 CRUD |
| **模型訓練** | Optuna 自動調參 + IC 去重複 + CSRankNorm |
| **Walk-Forward 回測** | 多模型滾動驗證，計算 IC Decay |
| **即時更新** | WebSocket + Zustand，CRUD 後自動刷新 UI |
| **預測推薦** | 選擇模型 + 日期，產生 Top-K 股票推薦 |

## 技術棧

| 層級 | 技術 |
|------|------|
| 後端 | FastAPI + SQLAlchemy + SQLite |
| 前端 | React 18 + Vite + Tailwind |
| 預測 | qlib + LightGBM + Optuna |
| 回測 | backtrader |
| 資料版本 | DVC + Google Drive |

## 快速開始

### 安裝

```bash
# 後端
pip install -r requirements.txt

# 前端
cd frontend && npm install
```

### 下載資料（DVC）

```bash
# 需要先安裝 Google Drive Desktop 並登入
python -m dvc pull
```

### 啟動

```bash
# 後端 (port 8000)
uvicorn src.interfaces.app:app --reload --port 8000

# 前端 (port 5173)
cd frontend && npm run dev
```

### 初始化因子

```bash
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## API 文檔

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 專案狀態

### 已完成

- [x] 資料同步（9 種資料集，TWSE/FinMind/yfinance）
- [x] Qlib 導出器（30 個欄位，PIT 月營收）
- [x] 因子管理（263 個預設因子，CRUD，驗證，去重複）
- [x] 前端 UI（8 個頁面）
- [x] 模型訓練（LightGBM，IC 增量選擇，Optuna 自動調參）
- [x] 即時更新（WebSocket + Zustand，CRUD 後自動刷新）
- [x] 回測系統（backtrader 整合，Equity Curve，績效指標，K-line 圖表）
- [x] Walk-Forward 回測（多模型滾動驗證，IC Decay 分析）
- [x] 預測推薦（選擇模型 + 日期，產生 Top-K 股票推薦）

### 待完成

- [ ] 增量學習（每日微調模型權重）
- [ ] 排程系統（每日自動同步+訓練）

## 資料來源

| 優先序 | 來源 | 說明 |
|--------|------|------|
| 1 | TWSE RWD | 官方資料，當日 17:30 後可用 |
| 2 | FinMind | 第三方整合，600次/時限制 |
| 3 | yfinance | 還原股價 |

## 文檔導覽

| 文檔 | 說明 |
|------|------|
| [API 設計](docs/api-design.md) | API 端點參考 |
| [訓練系統](docs/training-system.md) | 訓練流程、IC 計算、參數配置 |
| [資料集](docs/datasets.md) | 資料來源與更新時間 |
| [原始欄位](docs/raw-fields.md) | 30 個 Qlib 欄位定義 |
| [增量學習設計](docs/incremental-learning-design.md) | 未實現功能的設計文檔 |
| [模型績效分析](reports/model-performance-analysis.md) | 156 週 Walk-Forward 回測分析 |

## 授權

私人專案
