# qlib-tw-trader

台灣股票交易與預測系統，基於 qlib 量化框架。

## 技術棧

- **後端**: FastAPI + SQLAlchemy
- **預測引擎**: qlib
- **前端**: Vue 3（開發中）
- **資料庫**: SQLite

## 安裝

```bash
pip install -r requirements.txt
```

## 資料來源

- TWSE（台灣證券交易所）
- FinMind

## 開發

```bash
# 執行測試
pytest

# 啟動 API 服務
uvicorn main:app --reload
```

## 授權

私人專案
