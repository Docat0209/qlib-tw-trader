# Concept Drift 適應方案研究報告

> **最終決策**：採用 **方案 A：滑動窗口 + 週期重訓**
> - 訓練窗口：最近 2-3 年數據
> - 重訓頻率：每週或每月
> - 驗證期：1-2 週（縮短，減少 alpha 消耗）
>
> **理由**：保留 LightGBM、低記憶體需求、簡單實施，且學術研究顯示效果可能不輸複雜方案。

---

## 背景

### 當前系統問題

| 問題 | 數值 | 嚴重程度 |
|------|------|----------|
| IC 衰減（驗證→回測） | 89.2% | 嚴重 |
| 因子換手率 | 81% | 嚴重 |
| 回測 IC 顯著性 | p=0.42 | 不顯著 |

### 候選解決方案

| 方案 | 來源 | Base Model | 記憶體需求 |
|------|------|-----------|-----------|
| **DDG-DA** | Microsoft, AAAI 2022 | **不限制**（可用 LightGBM） | 官方 45GB（CSI300） |
| DoubleAdapt | SJTU, KDD 2023 | 需要神經網路（GRU） | ~10GB GPU |
| MetaDA | 2024 | 只支援神經網路 | 待確認 |
| LEAF | IJCAI 2025 | 只支援神經網路 | 待確認 |

---

## 核心原則：保留 LightGBM

### LightGBM vs GRU 在股票預測的表現

**學術結論（NeurIPS 2022）**：Tree-based 模型在 Tabular 資料上仍是業界最優方案。

| 特性 | LightGBM | GRU |
|------|----------|-----|
| Tabular 資料表現 | **業界最優** | 不如樹型 |
| 可解釋性 | 高（Feature Importance） | 黑盒 |
| 訓練速度 | 非常快 | 較慢 |
| 噪音處理 | 穩健 | 易被誤導 |

**結論**：不應該為了任何方法而放棄 LightGBM。需要找到能**保留 LightGBM** 的 Concept Drift 解決方案。

### 方案篩選標準

| 標準 | 必須 |
|------|------|
| 保留 LightGBM | 是 |
| 記憶體 ≤ 32GB | 是 |
| 有實際效果驗證 | 是 |
| 可與 Qlib 整合 | 優先 |

---

## DDG-DA 原理

### 核心創新

DDG-DA（Data Distribution Generation for Predictable Concept Drift Adaptation）：

1. **預測性適應**：提前預測數據分佈的演變（而非被動反應）
2. **分佈生成**：根據預測的未來分佈生成訓練樣本
3. **主動適應**：在 Concept Drift 發生前進行預防性調整

### 工作流程

```
Step 1: 訓練分佈預測器 → 估計未來數據分佈
   ↓
Step 2: 利用預測分佈生成/重加權訓練樣本
   ↓
Step 3: 在適應後的數據上訓練 LightGBM
   ↓
Step 4: 模型自動適應 Concept Drift
```

### 為什麼 DDG-DA 可以和 LightGBM 配合？

DDG-DA 的適應發生在**數據層**，不涉及模型內部：
- 不需要梯度反向傳播
- 不需要修改 LightGBM 架構
- 只是改變訓練數據的分佈/權重

---

## DDG-DA 記憶體問題分析

### 為什麼官方說需要 45GB？

DDG-DA 在 Qlib 的實現（`qlib/contrib/meta/data_selection/dataset.py`）有一個**已知的記憶體問題**：

```python
# 第 150 行註解
# FIXME: memory issue in this step
```

**記憶體爆炸的原因**：

1. **所有滾動任務同時載入記憶體**：
   ```python
   self.meta_task_l = []
   for t in task_iter:
       self.meta_task_l.append(MetaTaskDS(...))  # 60+ 個任務全部載入！
   ```

2. **每個 MetaTaskDS 儲存完整資料**：
   - `X`：訓練特徵矩陣（樣本數 × 因子數）
   - `time_belong`：樣本歸屬矩陣（樣本數 × 時段數）
   - `X_test`、`y`、`y_test` 等

3. **CSI300 規模估算**：
   - 300 股票 × 1250 天 = 375,000 樣本
   - 158 因子（Alpha158）
   - 60 個滾動任務

   每個任務特徵：375,000 × 158 × 4 bytes ≈ 237MB
   60 個任務：60 × 237MB ≈ 14GB（僅特徵）
   加上 time_belong、IC 矩陣、梯度 → **45GB**

### 台股規模估算

| 項目 | CSI300 | 台股（市值前 100） |
|------|--------|-------------------|
| 股票數 | 300 | 100 |
| 訓練期 | 5 年 | 3 年 |
| 樣本數 | 375,000 | 75,000 |
| 因子數 | 158 | 266 |
| 滾動任務 | 60 | 36 |
| **估計記憶體** | 45GB | **~15-20GB** |

**結論**：台股規模較小，理論上 **15-20GB** 可能足夠，但仍有 OOM 風險。

---

## MetaDA 和 LEAF 研究結果

### MetaDA (2024)

| 項目 | 結果 |
|------|------|
| 可用 LightGBM？ | **否**，只測試了 LSTM、GRU、Transformer |
| 有公開程式碼？ | **否**，論文說「補充材料中提供」但無連結 |
| Qlib 整合？ | 使用 Alpha360 特徵，但無整合範例 |

### LEAF (IJCAI 2025)

| 項目 | 結果 |
|------|------|
| 可用 LightGBM？ | **否**，「model-agnostic」只指深度學習模型 |
| 有公開程式碼？ | **否**，論文審查時標記「無可識別 URL」 |
| 記憶體需求？ | 未提及 |

### 結論

**MetaDA 和 LEAF 都不能保留 LightGBM**，都只適用於神經網路。

---

## Python 套件選項（可與 LightGBM 配合）

| 套件 | 功能 | GitHub Stars | 適用場景 |
|------|------|-------------|----------|
| **[River](https://github.com/online-ml/river)** | Online learning + drift 檢測 | 5k+ | 資料流、即時學習 |
| **[Alibi-Detect](https://github.com/SeldonIO/alibi-detect)** | Drift/Outlier 檢測 | 2k+ | 模型監控 |
| **[NannyML](https://nannyml.readthedocs.io/)** | Performance 估計 + drift 定位 | 2k+ | 精確定位 drift 時間點 |
| **[Evidently AI](https://github.com/evidentlyai/evidently)** | 通用 drift 監控 | 5k+ | 整合 Grafana/Prometheus |

### River 的 Drift 檢測方法
- **ADWIN**: Adaptive Windowing
- **DDM**: Drift Detection Method
- **EDDM**: Early Drift Detection Method

### 使用方式
這些套件用於**檢測 drift**，然後觸發 LightGBM 重訓：
```python
# 偽代碼
if drift_detector.detected():
    model = lgb.train(params, new_data, init_model=old_model)
```

---

## LightGBM 原生增量學習

LightGBM 支援使用 `init_model` 參數繼續訓練：

```python
# 第一次訓練
model1 = lgb.train(params, data1)
model1.save_model('model.txt')

# 增量訓練（在新數據上）
model2 = lgb.train(params, data2, init_model='model.txt')
```

**注意事項**（來自 [GitHub Issue #3781](https://github.com/microsoft/LightGBM/issues/3781)）：
> "在新數據上訓練後，舊數據的準確度可能會下降"

---

## 業界實踐（學術研究）

### MIT 研究：91% 的 ML 模型會退化
> [Temporal Quality Degradation in AI Models](https://www.nature.com/articles/s41598-022-15245-z)（Nature Scientific Reports）

### 重訓頻率建議

| 頻率 | 適用場景 |
|------|----------|
| **每日** | 高頻交易、即時競價 |
| **每週** | 多數系統的甜蜜點 |
| **每月** | 緩慢變化的系統 |

### 滾動窗口最佳實踐
> [HARd to Beat](https://arxiv.org/html/2406.08041v1)（2024）
> "Simple models with daily rolling windows achieve lower prediction error than complex ML models with static windows"

**建議窗口大小**：600-1000 天（2-3 年）

### 業界做法（來自學術論文）
1. **靜態窗口**：70% train / 10% valid / 20% test
2. **擴展窗口**：每 250 天更新一次（降低計算成本）
3. **Walk-Forward Optimization**：連續重訓

---

## 最終研究結論

### 核心發現

1. **DDG-DA 是唯一學術上驗證有效且可保留 LightGBM 的方案**，但需要 45GB 記憶體
2. **沒有找到 DDG-DA 的記憶體優化版本**
3. **業界實際做法**：Rolling Window + 定期重訓（比 DDG-DA 簡單很多）

### 為什麼業界不用複雜方案？

根據研究：
> "ML models fail to surpass the linear benchmark when utilizing a refined fitting approach"
> "Simple models with daily rolling windows achieve lower prediction error"

**結論**：複雜的 meta-learning 方案（DDG-DA、DoubleAdapt）可能**不比簡單的 rolling retrain 更有效**。

### 可行方案（按推薦順序）

| 優先級 | 方案 | 保留 LightGBM | 記憶體 | 複雜度 |
|--------|------|--------------|--------|--------|
| 1 | 滑動窗口 + 週期重訓 | 是 | 低 | 低 |
| 2 | Drift 檢測 + 觸發重訓 | 是 | 低 | 中 |
| 3 | LightGBM 增量學習（init_model）| 是 | 低 | 低 |
| 4 | 縮短驗證期 + Embargo | 是 | 低 | 低 |
| 5 | 自己實現簡化版 DDG-DA | 是 | 中 | 高 |

---

## 建議實施方案

### 方案 A：滑動窗口 + 週期重訓（已選擇）

```
訓練窗口：最近 2-3 年數據
重訓頻率：每週或每月
驗證期：1-2 週（縮短，減少 alpha 消耗）
```

### 方案 B：Drift 檢測 + 觸發重訓

使用 River 或 Evidently AI 檢測 drift：
```python
# 偽代碼
drift_detector = river.drift.ADWIN()
for new_data in data_stream:
    if drift_detector.update(prediction_error):
        model = retrain(recent_data)
```

### 方案 C：LightGBM 增量學習

使用 `init_model` 在新數據上微調：
```python
new_model = lgb.train(
    params,
    new_data,
    init_model=old_model,  # 繼續訓練
    num_boost_round=50     # 少量迭代
)
```

---

## 參考資源

### 核心論文
- [DDG-DA（AAAI 2022）](https://cdn.aaai.org/ojs/20327/20327-13-24340-1-2-20220628.pdf)
- [DoubleAdapt（KDD 2023）](https://arxiv.org/pdf/2306.09862)
- [MetaDA（2024）](https://arxiv.org/html/2401.03865v3)
- [LEAF（IJCAI 2025）](https://www.ijcai.org/proceedings/2025/0542.pdf)

### 技術參考
- [Qlib DDG-DA 實現](https://github.com/microsoft/qlib/tree/main/examples/benchmarks_dynamic/DDG-DA)
- [SJTU-DMTai/qlib fork](https://github.com/SJTU-DMTai/qlib)
- [DoubleAdapt 官方 API（無 Qlib 依賴）](https://github.com/SJTU-DMTai/DoubleAdapt)
- [Why Tree-Based Models Outperform Deep Learning（NeurIPS 2022）](https://arxiv.org/abs/2207.08815)
