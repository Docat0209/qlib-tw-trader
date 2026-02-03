# 增量學習與訓練配置設計文檔

## 背景問題

### 實驗結果（2026-02-03）

使用 Walk-Forward 多模型回測，驗證 IC vs 實盤 IC：

| 指標 | 數值 |
|------|------|
| 平均驗證期 IC | 0.2757 |
| 平均實盤期 IC | 0.0716 |
| **IC 衰減** | **76%** |
| IC 相關性 | 0.49 |

### 問題診斷

1. **驗證期 IC 高估**：0.28 的 IC 是「假象」，實盤只有 0.07
2. **Walk-Forward 無法解決**：即使每週重訓，IC 仍大幅衰減
3. **模型「過時」**：訓練完到上線交易，市場已經變化

---

## 研究發現

### Walk-Forward 的已知缺陷

來源：[Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

> "Walk forward testing is most common. However, **it is very easy to overfit because only 1 history is tested**. Thus it may not be a good indicator of future performance."

### 驗證後重訓是標準做法

來源：[The Alpha Scientist](https://alphascientist.com/walk_forward_model_building.html)

> "Before evaluating on the test set, **we retrain the model on the training data combined with the validation data** so that the model has seen the most recent observations."
>
> "**If we don't retrain the model this way, it will appear to underperform.**"

來源：Elements of Statistical Learning

> "Our final chosen model [after cross-validation] is f(x), which we then **fit to all the data**."

### 驗證期應匹配重訓頻率

來源：[DataRobot](https://docs.datarobot.com/en/docs/modeling/time/ts-adv-modeling/ts-customization.html)

> "**Validation length describes the frequency with which you retrain your models.** If you plan to retrain every 2 weeks, set validation length to 14 days."

來源：[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231217311074)

> "**The highest prediction performance is observed when the input window length is approximately equal to the forecast horizon.**"

---

## 解決方案

### 兩層架構

| 層級 | 決定什麼 | 更新頻率 | 方法 |
|------|----------|----------|------|
| **因子選擇** | 用哪些因子 | 每週 | IC 增量選擇，完整重訓 |
| **模型權重** | 因子怎麼組合 | 每日 | `init_model` 增量學習 |

### 核心原理

```
模型訓練 → 決定「選哪些因子」（結構）
增量學習 → 調整「因子權重」（參數）

因子結構變化慢（週/月級別）
因子權重變化快（日級別）
```

### LightGBM 增量學習

```python
# Day 1：完整訓練（含因子選擇）
model_v1 = lgb.train(params, train_data)

# Day 2：增量更新（因子不變，權重微調）
model_v2 = lgb.train(
    params,
    new_data,
    init_model=model_v1,      # 從 v1 開始
    keep_training_booster=True
)
```

---

## 配置參數

### 新配置（建議）

```python
# src/shared/constants.py

TRAIN_DAYS = 252      # 訓練期：1 年（原 504 天）
VALID_DAYS = 5        # 驗證期：1 週（原 10 天）
EMBARGO_DAYS = 5      # Embargo：1 週（維持）
RETRAIN_THRESHOLD_DAYS = 5  # 完整重訓週期：每週
```

### 配置對照

| 配置 | 訓練期 | 驗證期 | 預期 IC 衰減 |
|------|--------|--------|-------------|
| 舊配置 | 3 年 (756天) | 6 個月 (126天) | 嚴重 |
| 原配置 | 2 年 (504天) | 2 週 (10天) | 76% |
| **新配置** | **1 年 (252天)** | **1 週 (5天)** | < 50% |

### 為什麼這樣選？

**訓練期 1 年：**
- 太長（3年）：模型學到過時的 pattern
- 太短（3個月）：樣本不足，不穩定
- 1 年：平衡樣本量與時效性

**驗證期 1 週：**
- 匹配重訓頻率（每週）
- 研究支持：驗證期 ≈ 預測期

**Embargo 1 週：**
- 防止 label lookahead
- Label 定義需要 T+2 收盤價

---

## 完整工作流程

### 週日：完整訓練

```
1. 訓練期資料：T-257 ~ T-5（過去 1 年，扣除 embargo）
2. Embargo：T-5 ~ T（5 天緩衝）
3. 驗證期資料：T ~ T+5（1 週）

4. IC 增量選擇 → 選出 N 個因子
5. 訓練 LightGBM → 模型 A

6. 計算驗證期 IC（官方指標，用於評估模型品質）

7. 增量更新：A + 驗證期資料 → 模型 A'
   （標準做法：驗證後用全部資料重訓）

8. 保存 A' 準備部署
```

### 週一~週五：每日增量

```
1. 收盤後取得新資料
2. 計算新資料的 label（需等 T+2）
3. 增量更新：A' + 新資料 → A''

技術限制：
- T 日的 label = close(T+2) / close(T+1) - 1
- 所以增量更新有 2 天延遲
- 週一只能用到上週三的 label
```

### 時間線示意

```
週日訓練：
├─────────────────────┼─────────┼─────────┤
│   訓練期 (252天)     │ Embargo │  驗證期  │
│      1 年           │  5 天   │  5 天   │
├─────────────────────┼─────────┼─────────┤
                    T-5        T       T+5
                               ↑
                          驗證 IC 計算
                               ↓
                    增量更新後部署 A'

週一~週五：
T+6: 用 A' 交易，收盤後等待
T+8: 可計算 T+6 的 label，增量更新 → A''
T+9: 用 A'' 交易 ...
```

---

## 預期效果

### 改善來源

1. **縮短驗證期**：減少「過時」問題
2. **增量更新**：模型持續適應最新 pattern
3. **驗證後重訓**：標準做法，最大化資料利用

### 預期指標

| 指標 | 原本 | 預期 |
|------|------|------|
| 驗證 IC vs 實盤 IC 差距 | 76% | < 50% |
| 實盤 IC | 0.07 | 0.10+ |
| 模型適應速度 | 每週更新 | 每日微調 |

---

## 實作清單

### Phase 1：調整配置

- [ ] 修改 `constants.py`：TRAIN_DAYS=252, VALID_DAYS=5
- [ ] 修改訓練流程：驗證後增量更新再保存

### Phase 2：增量學習

- [ ] `ModelTrainer` 新增 `incremental_update()` 方法
- [ ] 支援 `init_model` 參數
- [ ] 每日排程呼叫增量更新

### Phase 3：驗證

- [ ] 訓練新模型（新配置）
- [ ] 執行 Walk-Forward 回測
- [ ] 比較 IC 衰減是否改善

---

## 參考文獻

1. [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos López de Prado
2. [Walk-Forward Model Building - The Alpha Scientist](https://alphascientist.com/walk_forward_model_building.html)
3. [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) - Hastie, Tibshirani, Friedman
4. [DoubleAdapt: Meta-learning for Incremental Learning](https://arxiv.org/pdf/2306.09862)
5. [DataRobot Time Series Customization](https://docs.datarobot.com/en/docs/modeling/time/ts-adv-modeling/ts-customization.html)
6. [Backtest Overfitting in the ML Era](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
7. [Alpha Decay - Di Mascio et al.](https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf)
