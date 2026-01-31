# qlib-tw-trader

台灣股票交易與預測系統，基於 Microsoft qlib 量化框架。

## 功能

- **資料同步**：自動抓取 TWSE、FinMind、yfinance 資料
- **Qlib 導出**：將資料轉換為 qlib .bin 格式
- **因子管理**：30 個預設因子，支援自訂與驗證
- **模型訓練**：IC 增量選擇法（開發中）
- **回測系統**：績效分析（開發中）

## 技術棧

| 層級 | 技術 |
|------|------|
| 後端 | FastAPI + SQLAlchemy |
| 前端 | React 18 + Vite + Tailwind |
| 預測 | qlib |
| 資料庫 | SQLite |

## 安裝

```bash
# 後端
pip install -r requirements.txt

# 前端
cd frontend && npm install
```

## 啟動

```bash
# 後端 (port 8000)
uvicorn src.interfaces.app:app --reload --port 8000

# 前端 (port 5173)
cd frontend && npm run dev
```

## API 文檔

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 資料來源

- TWSE RWD（台灣證券交易所）
- FinMind（第三方整合 API）
- yfinance（還原股價）

## 授權

私人專案
