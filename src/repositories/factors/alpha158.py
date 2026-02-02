"""
Alpha158 純 K 線因子

基於 Qlib Alpha158 因子集設計，使用純價格和成交量資料。
共計約 95 個因子。
"""

# 時間窗口
WINDOWS = [5, 10, 20, 30, 60]

# =============================================================================
# KBAR 類因子（9 個）
# K 線形態因子，描述單日 K 棒特徵
# =============================================================================

KBAR_FACTORS = [
    {
        "name": "kbar_kmid",
        "display_name": "K棒中位",
        "category": "technical",
        "expression": "($close - $open) / $open",
        "description": "收盤相對開盤的變化率",
    },
    {
        "name": "kbar_klen",
        "display_name": "K棒長度",
        "category": "technical",
        "expression": "($high - $low) / $open",
        "description": "日內振幅相對開盤價",
    },
    {
        "name": "kbar_kmid2",
        "display_name": "K棒中位2",
        "category": "technical",
        "expression": "($close - $open) / ($high - $low + 1e-8)",
        "description": "收盤開盤差佔振幅比",
    },
    {
        "name": "kbar_kup",
        "display_name": "上影線",
        "category": "technical",
        "expression": "($high - Greater($open, $close)) / $open",
        "description": "上影線長度比",
    },
    {
        "name": "kbar_kup2",
        "display_name": "上影線2",
        "category": "technical",
        "expression": "($high - Greater($open, $close)) / ($high - $low + 1e-8)",
        "description": "上影線佔振幅比",
    },
    {
        "name": "kbar_klow",
        "display_name": "下影線",
        "category": "technical",
        "expression": "(Less($open, $close) - $low) / $open",
        "description": "下影線長度比",
    },
    {
        "name": "kbar_klow2",
        "display_name": "下影線2",
        "category": "technical",
        "expression": "(Less($open, $close) - $low) / ($high - $low + 1e-8)",
        "description": "下影線佔振幅比",
    },
    {
        "name": "kbar_ksft",
        "display_name": "K棒偏移",
        "category": "technical",
        "expression": "(2 * $close - $high - $low) / $open",
        "description": "收盤價偏離中點比",
    },
    {
        "name": "kbar_ksft2",
        "display_name": "K棒偏移2",
        "category": "technical",
        "expression": "(2 * $close - $high - $low) / ($high - $low + 1e-8)",
        "description": "收盤價偏離中點佔振幅比",
    },
]


# =============================================================================
# ROC 類因子（5 個）
# 價格變化率
# =============================================================================

ROC_FACTORS = [
    {
        "name": f"roc_{w}",
        "display_name": f"ROC_{w}日",
        "category": "technical",
        "expression": f"Ref($close, {w}) / $close - 1",
        "description": f"{w}日價格變化率（歷史價/現價）",
    }
    for w in WINDOWS
]


# =============================================================================
# MA_RATIO 類因子（5 個）
# 均線比率
# =============================================================================

MA_RATIO_FACTORS = [
    {
        "name": f"ma_{w}",
        "display_name": f"均線比_{w}日",
        "category": "technical",
        "expression": f"Mean($close, {w}) / $close",
        "description": f"{w}日均線/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# STD 類因子（5 個）
# 波動率
# =============================================================================

STD_FACTORS = [
    {
        "name": f"std_{w}",
        "display_name": f"波動率_{w}日",
        "category": "technical",
        "expression": f"Std($close, {w}) / $close",
        "description": f"{w}日價格標準差/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# BETA 類因子（5 個）
# 價格趨勢斜率
# =============================================================================

BETA_FACTORS = [
    {
        "name": f"beta_{w}",
        "display_name": f"斜率_{w}日",
        "category": "technical",
        "expression": f"Slope($close, {w}) / $close",
        "description": f"{w}日價格回歸斜率/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# RSQR 類因子（5 個）
# 趨勢擬合度
# =============================================================================

RSQR_FACTORS = [
    {
        "name": f"rsqr_{w}",
        "display_name": f"R平方_{w}日",
        "category": "technical",
        "expression": f"Rsquare($close, {w})",
        "description": f"{w}日趨勢擬合度",
    }
    for w in WINDOWS
]


# =============================================================================
# RESI 類因子（5 個）
# 殘差
# =============================================================================

RESI_FACTORS = [
    {
        "name": f"resi_{w}",
        "display_name": f"殘差_{w}日",
        "category": "technical",
        "expression": f"Resi($close, {w}) / $close",
        "description": f"{w}日回歸殘差/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# MAX 類因子（5 個）
# 最高價比
# =============================================================================

MAX_FACTORS = [
    {
        "name": f"max_{w}",
        "display_name": f"最高價比_{w}日",
        "category": "technical",
        "expression": f"Max($high, {w}) / $close",
        "description": f"{w}日最高價/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# MIN 類因子（5 個）
# 最低價比
# =============================================================================

MIN_FACTORS = [
    {
        "name": f"min_{w}",
        "display_name": f"最低價比_{w}日",
        "category": "technical",
        "expression": f"Min($low, {w}) / $close",
        "description": f"{w}日最低價/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# QTLU 類因子（5 個）
# 80 分位數
# =============================================================================

QTLU_FACTORS = [
    {
        "name": f"qtlu_{w}",
        "display_name": f"80分位_{w}日",
        "category": "technical",
        "expression": f"Quantile($close, {w}, 0.8) / $close",
        "description": f"{w}日 80 分位數/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# QTLD 類因子（5 個）
# 20 分位數
# =============================================================================

QTLD_FACTORS = [
    {
        "name": f"qtld_{w}",
        "display_name": f"20分位_{w}日",
        "category": "technical",
        "expression": f"Quantile($close, {w}, 0.2) / $close",
        "description": f"{w}日 20 分位數/收盤價",
    }
    for w in WINDOWS
]


# =============================================================================
# RANK 類因子（5 個）
# 時序排名
# =============================================================================

RANK_FACTORS = [
    {
        "name": f"tsrank_{w}",
        "display_name": f"時序排名_{w}日",
        "category": "technical",
        "expression": f"Rank($close, {w})",
        "description": f"收盤價在 {w} 日內的排名",
    }
    for w in WINDOWS
]


# =============================================================================
# RSV 類因子（5 個）
# 相對強弱值
# =============================================================================

RSV_FACTORS = [
    {
        "name": f"rsv_{w}",
        "display_name": f"RSV_{w}日",
        "category": "technical",
        "expression": f"($close - Min($low, {w})) / (Max($high, {w}) - Min($low, {w}) + 1e-8)",
        "description": f"{w}日相對強弱值",
    }
    for w in WINDOWS
]


# =============================================================================
# CNTP 類因子（5 個）
# 上漲天數佔比
# =============================================================================

CNTP_FACTORS = [
    {
        "name": f"cntp_{w}",
        "display_name": f"上漲佔比_{w}日",
        "category": "technical",
        "expression": f"Mean(Greater($close - Ref($close, 1), 0), {w})",
        "description": f"{w}日上漲天數佔比",
    }
    for w in WINDOWS
]


# =============================================================================
# CNTN 類因子（5 個）
# 下跌天數佔比
# =============================================================================

CNTN_FACTORS = [
    {
        "name": f"cntn_{w}",
        "display_name": f"下跌佔比_{w}日",
        "category": "technical",
        "expression": f"Mean(Greater(Ref($close, 1) - $close, 0), {w})",
        "description": f"{w}日下跌天數佔比",
    }
    for w in WINDOWS
]


# =============================================================================
# SUMP 類因子（5 個）
# 上漲幅度佔比
# =============================================================================

SUMP_FACTORS = [
    {
        "name": f"sump_{w}",
        "display_name": f"上漲幅度_{w}日",
        "category": "technical",
        "expression": f"Sum(Greater($close - Ref($close, 1), 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-8)",
        "description": f"{w}日上漲幅度佔比",
    }
    for w in WINDOWS
]


# =============================================================================
# SUMN 類因子（5 個）
# 下跌幅度佔比
# =============================================================================

SUMN_FACTORS = [
    {
        "name": f"sumn_{w}",
        "display_name": f"下跌幅度_{w}日",
        "category": "technical",
        "expression": f"Sum(Greater(Ref($close, 1) - $close, 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-8)",
        "description": f"{w}日下跌幅度佔比",
    }
    for w in WINDOWS
]


# =============================================================================
# SUMD 類因子（5 個）
# 漲跌差異
# =============================================================================

SUMD_FACTORS = [
    {
        "name": f"sumd_{w}",
        "display_name": f"漲跌差_{w}日",
        "category": "technical",
        "expression": (
            f"Sum(Greater($close - Ref($close, 1), 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-8) "
            f"- Sum(Greater(Ref($close, 1) - $close, 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-8)"
        ),
        "description": f"{w}日漲跌幅度差異",
    }
    for w in WINDOWS
]


# =============================================================================
# VMA 類因子（5 個）
# 成交量均線比
# =============================================================================

VMA_FACTORS = [
    {
        "name": f"vma_{w}",
        "display_name": f"量均線比_{w}日",
        "category": "technical",
        "expression": f"Mean($volume, {w}) / ($volume + 1e-8)",
        "description": f"{w}日成交量均線/當日成交量",
    }
    for w in WINDOWS
]


# =============================================================================
# VSTD 類因子（5 個）
# 成交量波動率
# =============================================================================

VSTD_FACTORS = [
    {
        "name": f"vstd_{w}",
        "display_name": f"量波動_{w}日",
        "category": "technical",
        "expression": f"Std($volume, {w}) / (Mean($volume, {w}) + 1e-8)",
        "description": f"{w}日成交量變異係數",
    }
    for w in WINDOWS
]


# =============================================================================
# WVMA 類因子（5 個）
# 價量協方差
# =============================================================================

WVMA_FACTORS = [
    {
        "name": f"wvma_{w}",
        "display_name": f"價量協變_{w}日",
        "category": "technical",
        "expression": f"Corr($close, Log($volume + 1), {w})",
        "description": f"{w}日價格與成交量對數相關係數",
    }
    for w in WINDOWS
]


# =============================================================================
# 匯出所有 Alpha158 因子
# =============================================================================

ALPHA158_FACTORS = (
    KBAR_FACTORS
    + ROC_FACTORS
    + MA_RATIO_FACTORS
    + STD_FACTORS
    + BETA_FACTORS
    + RSQR_FACTORS
    + RESI_FACTORS
    + MAX_FACTORS
    + MIN_FACTORS
    + QTLU_FACTORS
    + QTLD_FACTORS
    + RANK_FACTORS
    + RSV_FACTORS
    + CNTP_FACTORS
    + CNTN_FACTORS
    + SUMP_FACTORS
    + SUMN_FACTORS
    + SUMD_FACTORS
    + VMA_FACTORS
    + VSTD_FACTORS
    + WVMA_FACTORS
)

# 因子數量統計
# KBAR: 9
# ROC: 5
# MA_RATIO: 5
# STD: 5
# BETA: 5
# RSQR: 5
# RESI: 5
# MAX: 5
# MIN: 5
# QTLU: 5
# QTLD: 5
# RANK: 5
# RSV: 5
# CNTP: 5
# CNTN: 5
# SUMP: 5
# SUMN: 5
# SUMD: 5
# VMA: 5
# VSTD: 5
# WVMA: 5
# 總計: 9 + 20 * 5 = 109 個
