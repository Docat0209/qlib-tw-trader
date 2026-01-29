# 資料來源最終規劃

## 設計原則

- 交易單位：**日**（排除即時/秒級資料）
- 資料策略：免費 API 優先 → 自行養資料最後

---

## 資料來源總覽

| 來源 | 類型 | 限制 |
|------|------|------|
| TWSE | 官方 | 5秒/3次，僅當日資料 |
| TPEX | 官方（上櫃） | 同上 |
| FinMind | 第三方 | 600次/時（免費） |
| yfinance | Yahoo Finance | 免費，有還原股價 |
| data.gov.tw | 政府開放資料 | 免費 |
| 國發會 | 景氣指標 | 免費下載 |
| TAIFEX | 期交所 | 免費 |

---

## 一、免費可用資料（排除即時類）

### 技術面（8 種）

| Dataset | 中文 | 優先來源 | 備用來源 |
|---------|------|----------|----------|
| TaiwanStockInfo | 台股總覽 | TWSE | FinMind |
| TaiwanStockInfoWithWarrant | 台股總覽(含權證) | TWSE | FinMind |
| TaiwanStockPrice | 日K線 | TWSE | FinMind |
| TaiwanStockPriceAdj | **還原股價** | **yfinance** ✅ | 自算 |
| TaiwanStockTradingDate | 交易日 | 從K線推算 | - |
| TaiwanStockIndustrialClassPrice | 類股股價 | TWSE | FinMind |
| TaiwanStockPER | PER/PBR | TWSE | FinMind |
| TaiwanStockDayTrading | 當沖成交量值 | TWSE | FinMind |
| TaiwanStockTotalReturnIndex | 報酬指數 | TWSE | FinMind |

**重大發現**：yfinance 提供 `Adj Close`（還原股價），免費且含歷史資料！

### 籌碼面（9 種）

| Dataset | 中文 | 優先來源 | 備用來源 |
|---------|------|----------|----------|
| TaiwanStockMarginPurchaseShortSale | 個股融資融券 | TWSE | FinMind |
| TaiwanStockTotalMarginPurchaseShortSale | 整體融資融券 | TWSE | FinMind |
| TaiwanStockInstitutionalInvestorsBuySell | 個股三大法人 | FinMind | TWSE(慢1-2天) |
| TaiwanStockTotalInstitutionalInvestors | 整體三大法人 | TWSE | FinMind |
| TaiwanStockShareholding | 外資持股 | TWSE | FinMind |
| TaiwanStockSecuritiesLending | 借券明細 | TWSE | FinMind |
| TaiwanStockMarginShortSaleSuspension | 融券回補日 | TWSE | FinMind |
| TaiwanDailyShortSaleBalances | 信用額度餘額 | TWSE | FinMind |
| TaiwanSecuritiesTraderInfo | 證券商資訊 | TWSE | FinMind |

### 基本面（10 種）

| Dataset | 中文 | 優先來源 | 備用來源 |
|---------|------|----------|----------|
| TaiwanStockCashFlowsStatement | 現金流量表 | FinMind | - |
| TaiwanStockFinancialStatements | 綜合損益表 | FinMind | - |
| TaiwanStockBalanceSheet | 資產負債表 | FinMind | - |
| TaiwanStockDividend | 股利政策 | FinMind | - |
| TaiwanStockDividendResult | 除權息結果 | TWSE | FinMind |
| TaiwanStockMonthRevenue | 月營收 | FinMind | - |
| TaiwanStockCapitalReductionReferencePrice | 減資參考價 | TWSE | FinMind |
| TaiwanStockDelisting | 下市資料 | TWSE | FinMind |
| TaiwanStockSplitPrice | 分割參考價 | TWSE | FinMind |
| TaiwanStockParValueChange | 變更面額參考價 | TWSE | FinMind |

### 衍生品（6 種）

| Dataset | 中文 | 來源 |
|---------|------|------|
| TaiwanFuturesDaily | 期貨日成交 | FinMind / TAIFEX |
| TaiwanOptionDaily | 選擇權日成交 | FinMind / TAIFEX |
| TaiwanFuturesInstitutionalInvestors | 期貨三大法人 | FinMind / TAIFEX |
| TaiwanOptionInstitutionalInvestors | 選擇權三大法人 | FinMind / TAIFEX |
| TaiwanFuturesDealerTradingVolumeDaily | 期貨券商交易 | FinMind / TAIFEX |
| TaiwanOptionDealerTradingVolumeDaily | 選擇權券商交易 | FinMind / TAIFEX |

### 其他（8 種）

| Dataset | 中文 | 來源 | 備註 |
|---------|------|------|------|
| TaiwanStockNews | 新聞 | FinMind | 獨有 |
| GoldPrice | 黃金價格 | FinMind | 獨有 |
| CrudeOilPrices | 原油 | FinMind | 獨有 |
| USStockPrice | 美股股價 | FinMind / yfinance | |
| TaiwanExchangeRate | 匯率 | FinMind | 獨有 |
| InterestRate | 央行利率 | FinMind | 獨有 |
| GovernmentBonds | 美債 | FinMind | 獨有 |
| TaiwanBusinessIndicator | **景氣燈號** | **國發會** ✅ | 有下載 |

---

## 二、原付費資料 → 替代方案

### 已找到免費替代 ✅

| 原付費 Dataset | 中文 | 替代方案 |
|----------------|------|----------|
| TaiwanStockPriceAdj | 還原股價 | **yfinance** `Adj Close` |
| TaiwanStockWeekPrice | 週K | 日K 彙總（自算） |
| TaiwanStockMonthPrice | 月K | 日K 彙總（自算） |
| TaiwanStock10Year | 十年線 | 日K MA(2500)（自算） |
| TaiwanStockMarketValue | 市值 | 股價×股本（自算） |
| TaiwanBusinessIndicator | 景氣燈號 | **國發會網站下載** |

### 需自己養資料（最低優先度）⏳

| Dataset | 中文 | TWSE 端點 | 說明 |
|---------|------|-----------|------|
| TaiwanStockHoldingSharesPer | 股權分級表 | 有 | 需每日抓取累積 |
| TaiwanStockTradingDailyReport | 分點資料 | 有 | 需每日抓取累積 |
| TaiwanStockWarrantTradingDailyReport | 權證分點 | 有 | 需每日抓取累積 |
| TaiwanTotalExchangeMarginMaintenance | 融資維持率 | 有 | 需每日抓取累積 |
| TaiwanStockTradingDailyReportSecIdAgg | 券商分點統計 | 有 | 需每日抓取累積 |
| TaiwanStockDispositionSecuritiesPeriod | 處置公告 | 有 | 需每日抓取累積 |
| TaiwanStockSuspended | 暫停交易公告 | 有 | 需每日抓取累積 |
| TaiwanStockDayTradingSuspension | 當沖預告 | 有 | 需每日抓取累積 |
| TaiwanStockInfoWithWarrantSummary | 權證對照表 | 有 | 需每日抓取累積 |
| TaiwanStockMarketValueWeight | 市值比重 | 有 | 需每日抓取累積 |
| TaiwanFuturesInstitutionalInvestorsAfterHours | 期貨夜盤法人 | TAIFEX | 需每日抓取累積 |
| TaiwanOptionInstitutionalInvestorsAfterHours | 選擇權夜盤法人 | TAIFEX | 需每日抓取累積 |
| TaiwanFuturesOpenInterestLargeTraders | 期貨大額未沖銷 | TAIFEX | 需每日抓取累積 |
| TaiwanOptionOpenInterestLargeTraders | 選擇權大額未沖銷 | TAIFEX | 需每日抓取累積 |

### 無法取得（放棄）❌

| Dataset | 中文 | 原因 |
|---------|------|------|
| TaiwanstockGovernmentBankBuySell | 八大行庫 | 第三方彙整，非官方資料，無公開 API |
| TaiwanStockIndustryChain | 產業鏈 | FinMind 付費獨有 |

---

## 三、最終統計

| 分類 | 免費API | 自算 | 需養資料 | 放棄 | 總計 |
|------|---------|------|----------|------|------|
| 技術面 | 8 | 4 | 5 | 0 | 17 |
| 籌碼面 | 9 | 0 | 5 | 1 | 15 |
| 基本面 | 10 | 0 | 0 | 0 | 10 |
| 衍生品 | 6 | 0 | 4 | 0 | 10 |
| 其他 | 8 | 0 | 0 | 1 | 9 |
| **總計** | **41** | **4** | **14** | **2** | **61** |

---

## 四、Adapter 實作規劃

| Adapter | 負責資料 | 狀態 |
|---------|----------|------|
| TwseAdapter | 日K、融資融券、PER 等 | ✅ 已有日K |
| FinMindAdapter | 歷史補全、財報、三大法人 | ✅ 已有日K |
| **YfinanceAdapter** | **還原股價、美股** | ⏳ **新增** |
| TaifexAdapter | 期貨選擇權 | ⏳ 待開發 |
| NdcAdapter | 景氣燈號 | ⏳ 待開發 |
| CalculatedService | 週K、月K、十年線、市值 | ⏳ 待開發 |

---

## 五、實作優先順序

### Phase 1：核心資料（免費 API）
1. 日K線 - TWSE → FinMind 補全
2. **還原股價 - yfinance** ✅ 新增
3. 三大法人 - FinMind（TWSE 慢 1-2 天）
4. 融資融券 - TWSE
5. PER/PBR - TWSE
6. 除權息結果 - TWSE

### Phase 2：擴充資料（免費 API）
7. 外資持股、借券
8. 月營收、財報
9. 期貨選擇權
10. 黃金、原油、匯率、美股
11. **景氣燈號 - 國發會** ✅ 新增

### Phase 3：自算資料
12. 週K、月K（從日K彙總）
13. 十年線（MA2500）
14. 市值（股價×股本）

### Phase 4：自己養資料（最低優先度）
15. 分點資料
16. 股權分級表
17. 期貨大額未沖銷
18. 其他...

---

## 六、參考資料

| 來源 | 網址 |
|------|------|
| TWSE OpenAPI | https://openapi.twse.com.tw/ |
| TPEX OpenAPI | https://www.tpex.org.tw/openapi/ |
| FinMind | https://finmind.github.io/ |
| yfinance | https://github.com/ranaroussi/yfinance |
| 國發會景氣指標 | https://index.ndc.gov.tw/n/zh_tw/data/eco |
| data.gov.tw | https://data.gov.tw/ |
| TAIFEX | https://www.taifex.com.tw/ |
