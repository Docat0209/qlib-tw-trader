# FinMind Dataset ↔ TWSE/TAIFEX API 完整對應表

測試時間：2026-01-29 19:01（收盤後）

---

## 重要發現

1. **TWSE 有兩套 API**：
   - **OpenAPI** (`openapi.twse.com.tw`)：更新較慢，可能有數小時延遲
   - **RWD API** (`www.twse.com.tw/rwd`)：更新較快，收盤後即時

2. **建議使用 RWD API** 取得即時資料，OpenAPI 用於備援

3. **更新時間實測**（1/29 當天）：
   | 資料類型 | 預計時間 | 19:01 狀態 |
   |----------|----------|------------|
   | 日K線 (RWD) | ~15:00 | ✅ 有 1/29 |
   | 日K線 (OpenAPI) | ~15:00 | ❌ 還在 1/28 |
   | 三大法人 | ~16:30 | ✅ 有 1/29 |
   | 融資融券 | ~19:00 | ❌ 還在 1/28 |

---

## 一、每日更新資料

### 技術面

| FinMind Dataset | 中文名稱 | TWSE API | 端點 | 更新時間 | 1/29 狀態 |
|-----------------|----------|----------|------|----------|-----------|
| TaiwanStockPrice | 日K線 | RWD | `/rwd/zh/afterTrading/STOCK_DAY_ALL` | ~15:00 | ✅ |
| TaiwanStockPrice | 日K線 | OpenAPI | `/v1/exchangeReport/STOCK_DAY_ALL` | ~15:00+ | ❌ 延遲 |
| TaiwanStockPER | PER/PBR | RWD | `/rwd/zh/afterTrading/BWIBBU_ALL` | ~15:00 | ✅ (推測) |
| TaiwanStockPER | PER/PBR | OpenAPI | `/v1/exchangeReport/BWIBBU_ALL` | ~15:00+ | ❌ 延遲 |
| TaiwanStockPriceAdj | 還原股價 | **yfinance** | `{stock_id}.TW` | ~15:30 | ✅ |
| TaiwanStockDayTrading | 當沖 | RWD | `/rwd/zh/afterTrading/BWISU` | ~15:30 | ⚠️ 待確認 |

### 籌碼面

| FinMind Dataset | 中文名稱 | TWSE API | 端點 | 更新時間 | 1/29 狀態 |
|-----------------|----------|----------|------|----------|-----------|
| TaiwanStockInstitutionalInvestorsBuySell | 個股三大法人 | RWD | `/rwd/zh/fund/T86?selectType=ALL` | **~16:30** | ✅ |
| TaiwanStockTotalInstitutionalInvestors | 整體三大法人 | RWD | `/rwd/zh/fund/TWT38U` | **~16:30** | ✅ |
| TaiwanStockMarginPurchaseShortSale | 融資融券 | RWD | `/rwd/zh/marginTrading/MI_MARGN` | **~19:00+** | ❌ 延遲 |
| TaiwanStockShareholding | 外資持股 | RWD | `/rwd/zh/fund/MI_QFIIS` | **T+1** | ❌ 隔天 |
| TaiwanStockSecuritiesLending | 借券明細 | RWD | `/rwd/zh/lending/TWT93U` | ~17:00 | ⚠️ 待確認 |

### 衍生品（期交所）

| FinMind Dataset | 中文名稱 | 來源 | data_id | 更新時間 | 1/29 狀態 |
|-----------------|----------|------|---------|----------|-----------|
| TaiwanFuturesDaily | 期貨日成交 | FinMind/TAIFEX | TX | ~15:00 | ✅ |
| TaiwanOptionDaily | 選擇權日成交 | FinMind/TAIFEX | TXO | ~15:00 | ✅ |
| TaiwanFuturesInstitutionalInvestors | 期貨三大法人 | FinMind/TAIFEX | TX | ~15:00 | ✅ |
| TaiwanOptionInstitutionalInvestors | 選擇權三大法人 | FinMind/TAIFEX | TXO | ~15:00 | ✅ |

### 其他

| FinMind Dataset | 中文名稱 | 來源 | data_id | 更新時間 | 1/29 狀態 |
|-----------------|----------|------|---------|----------|-----------|
| TaiwanExchangeRate | 匯率 | FinMind | USD | 每日 | ✅ |
| GoldPrice | 黃金價格 | FinMind | - | 每日 | ✅ |
| CrudeOilPrices | 原油價格 | FinMind | WTI | **延遲 5-10 天** | ❌ 1/20 |

---

## 二、非每日更新資料

### 定期更新

| FinMind Dataset | 中文名稱 | 更新頻率 | 最新日期 | 說明 |
|-----------------|----------|----------|----------|------|
| TaiwanStockMonthRevenue | 月營收 | 每月10日前 | 2026-01-01 | 12月營收在1月公布 |
| TaiwanStockFinancialStatements | 綜合損益表 | 季報 | - | Q4 約 3 月底前公布 |
| TaiwanStockBalanceSheet | 資產負債表 | 季報 | - | 同上 |
| TaiwanStockCashFlowsStatement | 現金流量表 | 季報 | - | 同上 |
| TaiwanStockDividend | 股利政策 | 年度 | - | 股東會後公布 |

### 事件型更新

| FinMind Dataset | 中文名稱 | 觸發條件 | 說明 |
|-----------------|----------|----------|------|
| TaiwanStockDividendResult | 除權息結果 | 除權息日 | 有除權息才有資料 |
| TaiwanStockCapitalReductionReferencePrice | 減資參考價 | 減資日 | 有減資才有資料 |

---

## 三、TWSE API 格式說明

### RWD API 格式

```json
{
  "stat": "OK",
  "date": "20260129",
  "title": "...",
  "fields": ["欄位1", "欄位2", ...],
  "data": [
    ["值1", "值2", ...],
    ...
  ]
}
```

### OpenAPI 格式

```json
[
  {
    "Date": "1150129",
    "Code": "2330",
    "Name": "台積電",
    "OpeningPrice": "1790.00",
    ...
  },
  ...
]
```

### 日期格式轉換

| 格式 | 範例 | 說明 |
|------|------|------|
| 民國年 7 位 | `1150129` | OpenAPI 使用 |
| 西元年 8 位 | `20260129` | RWD API 使用 |
| ISO 格式 | `2026-01-29` | FinMind 使用 |

轉換函式：
```python
def roc_to_iso(roc: str) -> str:
    """民國年轉 ISO"""
    if len(roc) == 7:
        year = int(roc[:3]) + 1911
        return f"{year}-{roc[3:5]}-{roc[5:7]}"
    elif len(roc) == 8:
        return f"{roc[:4]}-{roc[4:6]}-{roc[6:8]}"
    return roc
```

---

## 四、TWSE RWD API 端點完整清單

### 交易資訊

| 端點 | 說明 | 對應 FinMind |
|------|------|--------------|
| `/rwd/zh/afterTrading/STOCK_DAY_ALL` | 每日收盤行情（全部） | TaiwanStockPrice |
| `/rwd/zh/afterTrading/BWIBBU_ALL` | PER/PBR（全部） | TaiwanStockPER |
| `/rwd/zh/afterTrading/BWISU` | 當沖資訊 | TaiwanStockDayTrading |

### 籌碼資訊

| 端點 | 說明 | 對應 FinMind |
|------|------|--------------|
| `/rwd/zh/fund/T86?selectType=ALL` | 三大法人買賣超（全部） | TaiwanStockInstitutionalInvestorsBuySell |
| `/rwd/zh/fund/TWT38U` | 三大法人彙總 | TaiwanStockTotalInstitutionalInvestors |
| `/rwd/zh/marginTrading/MI_MARGN` | 融資融券餘額 | TaiwanStockMarginPurchaseShortSale |
| `/rwd/zh/fund/MI_QFIIS` | 外資持股比例 | TaiwanStockShareholding |
| `/rwd/zh/lending/TWT93U` | 借券餘額 | TaiwanStockSecuritiesLending |

### 其他

| 端點 | 說明 | 對應 FinMind |
|------|------|--------------|
| `/rwd/zh/exRight/TWT49U` | 除權息結果 | TaiwanStockDividendResult |

---

## 五、資料取得策略

### 每日更新流程

```
15:00 後：
├── 日K線 (TWSE RWD)
├── PER/PBR (TWSE RWD)
├── 還原股價 (yfinance)
└── 期貨選擇權 (FinMind)

16:30 後：
└── 三大法人 (TWSE RWD)

19:00+ 後：
└── 融資融券 (TWSE RWD)

隔天：
└── 外資持股 (TWSE RWD)
```

### API 選擇優先序

1. **TWSE RWD API**：即時性最佳
2. **FinMind API**：歷史資料完整
3. **TWSE OpenAPI**：備援（有延遲）
4. **yfinance**：還原股價專用

---

## 六、實測結論

| 類別 | 1/29 可用 | 1/29 延遲 | 非每日 |
|------|-----------|-----------|--------|
| 技術面 | 2 | 2 | 0 |
| 籌碼面 | 2 | 3 | 0 |
| 基本面 | 0 | 0 | 6 |
| 衍生品 | 4 | 0 | 0 |
| 其他 | 2 | 1 | 0 |
| **總計** | **10** | **6** | **6** |

**每日資料**：16 種中有 10 種在 19:01 可用（62.5%）
**延遲資料**：融資融券、外資持股等需等更晚或隔天
