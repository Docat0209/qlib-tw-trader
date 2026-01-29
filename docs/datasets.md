# Dataset 資料來源總表

## 資料來源

| 來源 | 說明 | 限制 |
|------|------|------|
| TWSE OpenAPI | 證交所官方 | 5秒/3次，僅當日 |
| FinMind | 第三方整合 | 600次/時（免費） |
| yfinance | Yahoo Finance | 免費，有還原股價 |
| TAIFEX | 期交所（經 FinMind） | 同 FinMind |

## 更新時間（實測 2026-01-29）

| 資料 | 更新時間 | 備註 |
|------|----------|------|
| 日K線 | ~15:00 | |
| 還原股價 | ~15:30 | yfinance |
| 三大法人 | ~16:30 | |
| 期貨選擇權 | ~15:00 | FinMind |
| 融資融券 | ~21:00+ | 較晚 |
| 外資持股 | T+1 | 隔天 |

---

## 一、已可用 API 取得（26 種）

### 技術面

| FinMind Dataset | 中文名稱 | 來源 | 備註 |
|-----------------|----------|------|------|
| TaiwanStockInfo | 台股總覽 | TWSE | 從日K推算 |
| TaiwanStockPrice | 日K線 | TWSE/FinMind | |
| TaiwanStockPriceAdj | 還原股價 | yfinance | Adj Close |
| TaiwanStockPER | PER/PBR/殖利率 | TWSE/FinMind | |
| TaiwanStockDayTrading | 當沖成交量值 | TWSE | |
| TaiwanStockTotalReturnIndex | 報酬指數 | TWSE | |

### 籌碼面

| FinMind Dataset | 中文名稱 | 來源 | 備註 |
|-----------------|----------|------|------|
| TaiwanStockMarginPurchaseShortSale | 個股融資融券 | TWSE/FinMind | |
| TaiwanStockTotalMarginPurchaseShortSale | 整體融資融券 | TWSE | |
| TaiwanStockInstitutionalInvestorsBuySell | 個股三大法人 | TWSE/FinMind | |
| TaiwanStockTotalInstitutionalInvestors | 整體三大法人 | TWSE | |
| TaiwanStockShareholding | 外資持股 | TWSE/FinMind | T+1 |
| TaiwanStockSecuritiesLending | 借券明細 | TWSE/FinMind | |
| TaiwanDailyShortSaleBalances | 信用額度餘額 | TWSE/FinMind | |
| TaiwanSecuritiesTraderInfo | 證券商資訊 | TWSE | 靜態 |

### 基本面

| FinMind Dataset | 中文名稱 | 來源 | 備註 |
|-----------------|----------|------|------|
| TaiwanStockCashFlowsStatement | 現金流量表 | FinMind | 季報 |
| TaiwanStockFinancialStatements | 綜合損益表 | FinMind | 季報 |
| TaiwanStockBalanceSheet | 資產負債表 | FinMind | 季報 |
| TaiwanStockDividend | 股利政策 | FinMind | 年度 |
| TaiwanStockDividendResult | 除權息結果 | TWSE/FinMind | 事件型 |
| TaiwanStockMonthRevenue | 月營收 | FinMind | 每月10日前 |
| TaiwanStockCapitalReductionReferencePrice | 減資參考價 | TWSE/FinMind | 事件型 |
| TaiwanStockDelisting | 下市資料 | TWSE/FinMind | 事件型 |
| TaiwanStockSplitPrice | 分割參考價 | TWSE/FinMind | 事件型 |
| TaiwanStockParValueChange | 變更面額參考價 | TWSE/FinMind | 事件型 |

### 衍生品

| FinMind Dataset | 中文名稱 | 來源 | 備註 |
|-----------------|----------|------|------|
| TaiwanFuturesDaily | 期貨日成交 | FinMind | data_id=TX |
| TaiwanOptionDaily | 選擇權日成交 | FinMind | data_id=TXO |
| TaiwanFuturesInstitutionalInvestors | 期貨三大法人 | FinMind | |
| TaiwanOptionInstitutionalInvestors | 選擇權三大法人 | FinMind | |
| TaiwanOptionFutureInfo | 期貨選擇權總覽 | FinMind | |

### 其他

| FinMind Dataset | 中文名稱 | 來源 | 備註 |
|-----------------|----------|------|------|
| GoldPrice | 黃金價格 | FinMind | |
| CrudeOilPrices | 原油價格 | FinMind | 延遲 ~9 天 |
| TaiwanExchangeRate | 匯率 | FinMind | data_id=USD |
| TaiwanBusinessIndicator | 景氣燈號 | 國發會 | 待實作下載 |

---

## 二、需自行累積（17 種）

這些資料 TWSE 有端點，但 FinMind 免費版無歷史，需每日抓取累積。

### 籌碼面

| FinMind Dataset | 中文名稱 | TWSE 端點 |
|-----------------|----------|-----------|
| TaiwanStockHoldingSharesPer | 股權分級表 | 有 |
| TaiwanStockTradingDailyReport | 分點資料 | 有 |
| TaiwanStockWarrantTradingDailyReport | 權證分點 | 有 |
| TaiwanTotalExchangeMarginMaintenance | 融資維持率 | 有 |
| TaiwanStockTradingDailyReportSecIdAgg | 券商分點統計 | 有 |
| TaiwanStockDispositionSecuritiesPeriod | 處置公告 | 有 |

### 技術面

| FinMind Dataset | 中文名稱 | TWSE 端點 |
|-----------------|----------|-----------|
| TaiwanStockSuspended | 暫停交易公告 | 有 |
| TaiwanStockDayTradingSuspension | 當沖預告 | 有 |
| TaiwanStockInfoWithWarrant | 含權證總覽 | 有 |
| TaiwanStockInfoWithWarrantSummary | 權證對照表 | 有 |
| TaiwanStockMarketValueWeight | 市值比重 | 有 |
| TaiwanStockMarginShortSaleSuspension | 融券回補日 | 有 |

### 衍生品

| FinMind Dataset | 中文名稱 | 來源 |
|-----------------|----------|------|
| TaiwanFuturesInstitutionalInvestorsAfterHours | 期貨夜盤法人 | TAIFEX |
| TaiwanOptionInstitutionalInvestorsAfterHours | 選擇權夜盤法人 | TAIFEX |
| TaiwanFuturesDealerTradingVolumeDaily | 期貨券商交易 | TAIFEX |
| TaiwanOptionDealerTradingVolumeDaily | 選擇權券商交易 | TAIFEX |
| TaiwanFuturesOpenInterestLargeTraders | 期貨大額未沖銷 | TAIFEX |
| TaiwanOptionOpenInterestLargeTraders | 選擇權大額未沖銷 | TAIFEX |

---

## 三、等待實作（8 種）

### 可自算（從現有資料推導）

| FinMind Dataset | 中文名稱 | 計算方式 |
|-----------------|----------|----------|
| TaiwanStockWeekPrice | 週K線 | 日K 彙總 |
| TaiwanStockMonthPrice | 月K線 | 日K 彙總 |
| TaiwanStock10Year | 十年線 | MA(2500) |
| TaiwanStockMarketValue | 市值 | 股價 × 股本 |
| TaiwanStockTradingDate | 交易日 | 從日K推算 |

### 可轉債（優先度低）

| FinMind Dataset | 中文名稱 | 備註 |
|-----------------|----------|------|
| TaiwanStockConvertibleBondInfo | 可轉債總覽 | FinMind 有 |
| TaiwanStockConvertibleBondDaily | 可轉債日成交 | FinMind 有 |
| TaiwanStockConvertibleBondInstitutionalInvestors | 可轉債三大法人 | FinMind 有 |

---

## TWSE OpenAPI 端點對應

| FinMind Dataset | OpenAPI 端點 |
|-----------------|--------------|
| TaiwanStockPrice | `/v1/exchangeReport/STOCK_DAY_ALL` |
| TaiwanStockPER | `/v1/exchangeReport/BWIBBU_ALL` |
| TaiwanStockInstitutionalInvestorsBuySell | `/v1/exchangeReport/TWT43U_ALL` |
| TaiwanStockMarginPurchaseShortSale | `/v1/exchangeReport/MI_MARGN_ALL` |
| TaiwanStockShareholding | `/v1/exchangeReport/MI_QFIIS_ALL` |

Base URL: `https://openapi.twse.com.tw`

---

## 開盤前檢查

```bash
# 檢查前一交易日資料完整度
python sandbox/check_previous_day.py

# 指定日期
python sandbox/check_previous_day.py 2026-01-29
```

關鍵資料（必須有才能交易）：
- 日K線
- PER/PBR
- 三大法人
- 融資融券
- 還原股價
