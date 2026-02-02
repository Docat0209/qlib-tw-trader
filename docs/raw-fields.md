# 原始資料欄位

本文檔記錄系統中可用於因子計算的原始資料欄位。

## 欄位總覽

共 **30 個欄位**，分佈於 8 個資料表。

| 類別 | 欄位數 | 資料表 |
|------|--------|--------|
| OHLCV | 5 | stock_daily |
| 還原股價 | 1 | stock_daily_adj |
| 估值 | 3 | stock_daily_per |
| 三大法人 | 6 | stock_daily_institutional |
| 融資融券 | 6 | stock_daily_margin |
| 外資持股 | 7 | stock_daily_shareholding |
| 借券 | 1 | stock_daily_securities_lending |
| 月營收 | 1 | stock_monthly_revenue |

---

## 欄位詳細說明

### OHLCV（5 個）

日 K 線基本資料，來源：TWSE RWD / FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| open | `$open` | Numeric(10,2) | 開盤價 |
| high | `$high` | Numeric(10,2) | 最高價 |
| low | `$low` | Numeric(10,2) | 最低價 |
| close | `$close` | Numeric(10,2) | 收盤價 |
| volume | `$volume` | Integer | 成交量（張） |

### 還原股價（1 個）

考慮除權息調整的股價，來源：yfinance。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| adj_close | `$adj_close` | Numeric(10,2) | 還原收盤價 |

### 估值指標（3 個）

來源：TWSE RWD / FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| pe_ratio | `$pe_ratio` | Numeric(10,2) | 本益比 |
| pb_ratio | `$pb_ratio` | Numeric(10,2) | 股價淨值比 |
| dividend_yield | `$dividend_yield` | Numeric(10,4) | 殖利率 (%) |

### 三大法人買賣超（6 個）

來源：TWSE RWD / FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| foreign_buy | `$foreign_buy` | Integer | 外資買進（張） |
| foreign_sell | `$foreign_sell` | Integer | 外資賣出（張） |
| trust_buy | `$trust_buy` | Integer | 投信買進（張） |
| trust_sell | `$trust_sell` | Integer | 投信賣出（張） |
| dealer_buy | `$dealer_buy` | Integer | 自營商買進（張） |
| dealer_sell | `$dealer_sell` | Integer | 自營商賣出（張） |

### 融資融券（6 個）

來源：TWSE RWD / FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| margin_buy | `$margin_buy` | Integer | 融資買進（張） |
| margin_sell | `$margin_sell` | Integer | 融資賣出（張） |
| margin_balance | `$margin_balance` | Integer | 融資餘額（張） |
| short_buy | `$short_buy` | Integer | 融券買進（張） |
| short_sell | `$short_sell` | Integer | 融券賣出（張） |
| short_balance | `$short_balance` | Integer | 融券餘額（張） |

### 外資持股（7 個）

來源：TWSE RWD / FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| total_shares | `$total_shares` | Integer | 發行股數 |
| foreign_shares | `$foreign_shares` | Integer | 外資持股數 |
| foreign_ratio | `$foreign_ratio` | Numeric(6,2) | 外資持股比率 (%) |
| foreign_remaining_shares | `$foreign_remaining_shares` | Integer | 外資尚可投資股數 |
| foreign_remaining_ratio | `$foreign_remaining_ratio` | Numeric(6,2) | 外資尚可投資比率 (%) |
| foreign_upper_limit_ratio | `$foreign_upper_limit_ratio` | Numeric(6,2) | 外資投資上限 (%) |
| chinese_upper_limit_ratio | `$chinese_upper_limit_ratio` | Numeric(6,2) | 陸資投資上限 (%) |

### 借券（1 個）

來源：FinMind。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| lending_volume | `$lending_volume` | Integer | 借券成交量（張） |

### 月營收（1 個）

來源：FinMind，以 PIT（Point In Time）格式展開至日頻。

| 欄位名 | Qlib 名稱 | 型別 | 說明 |
|--------|-----------|------|------|
| revenue | `$revenue` | Numeric(16,0) | 月營收（千元） |

**注意**：N 月營收於 N+1 月 10 日前公布，系統會自動 forward fill 至下一次公布日。

---

## Qlib Expression 使用範例

```python
# 技術面
"$close / Ref($close, 5) - 1"           # 5日動能
"Mean($close, 5) / Mean($close, 20)"    # 均線比

# 籌碼面
"$foreign_buy - $foreign_sell"          # 外資淨買
"Sum($trust_buy - $trust_sell, 5)"      # 投信5日淨買

# 估值面
"1 / ($pe_ratio + 1e-8)"                # 益本比

# 營收面
"$revenue / Ref($revenue, 21) - 1"      # 營收月增
```

---

## 資料更新時間

| 資料 | 更新時間 | 延遲 |
|------|---------|------|
| 日 K 線 | ~15:00 | 即時 |
| 三大法人 | ~16:30 | T+0 |
| 融資融券 | ~21:00 | T+0 |
| 外資持股 | T+1 | T+1 |
| 借券 | ~16:30 | T+0 |
| 月營收 | 次月10日前 | 月度 |
