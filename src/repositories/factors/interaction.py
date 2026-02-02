"""
交互因子

跨類別組合因子，包含價量、籌碼價格、估值動能等交互關係。
共計約 55 個因子。
"""

# =============================================================================
# 價量交互因子
# =============================================================================

# 量價相關性
VOL_PRICE_CORR_FACTORS = [
    {
        "name": "vol_price_corr_5",
        "display_name": "量價相關_5日",
        "category": "interaction",
        "expression": "Corr($close, Log($volume + 1), 5)",
        "description": "5 日價量相關性",
    },
    {
        "name": "vol_price_corr_10",
        "display_name": "量價相關_10日",
        "category": "interaction",
        "expression": "Corr($close, Log($volume + 1), 10)",
        "description": "10 日價量相關性",
    },
    {
        "name": "vol_price_corr_20",
        "display_name": "量價相關_20日",
        "category": "interaction",
        "expression": "Corr($close, Log($volume + 1), 20)",
        "description": "20 日價量相關性",
    },
    {
        "name": "vol_price_corr_60",
        "display_name": "量價相關_60日",
        "category": "interaction",
        "expression": "Corr($close, Log($volume + 1), 60)",
        "description": "60 日價量相關性",
    },
]

# 收益與量變相關
RET_VOL_CORR_FACTORS = [
    {
        "name": "ret_vol_corr_5",
        "display_name": "收益量變相關_5日",
        "category": "interaction",
        "expression": "Corr($close / Ref($close, 1) - 1, $volume / (Ref($volume, 1) + 1e-8) - 1, 5)",
        "description": "5 日收益與量變相關性",
    },
    {
        "name": "ret_vol_corr_20",
        "display_name": "收益量變相關_20日",
        "category": "interaction",
        "expression": "Corr($close / Ref($close, 1) - 1, $volume / (Ref($volume, 1) + 1e-8) - 1, 20)",
        "description": "20 日收益與量變相關性",
    },
]

# 量能分佈
VOL_DISTRIBUTION_FACTORS = [
    {
        "name": "vol_up_down_ratio",
        "display_name": "上漲量能比",
        "category": "interaction",
        "expression": "Sum($volume * Greater($close - Ref($close, 1), 0), 20) / (Sum($volume * Greater(Ref($close, 1) - $close, 0), 20) + 1e-8)",
        "description": "20 日上漲量 / 下跌量",
    },
    {
        "name": "vol_weighted_ret",
        "display_name": "量加權收益",
        "category": "interaction",
        "expression": "Sum(($close / Ref($close, 1) - 1) * $volume, 20) / (Sum($volume, 20) + 1e-8)",
        "description": "20 日量加權平均收益",
    },
]

# OBV 類
OBV_FACTORS = [
    {
        "name": "obv_5",
        "display_name": "OBV_5日",
        "category": "interaction",
        "expression": "Sum($volume * Sign($close - Ref($close, 1)), 5)",
        "description": "5 日能量潮",
    },
    {
        "name": "obv_20",
        "display_name": "OBV_20日",
        "category": "interaction",
        "expression": "Sum($volume * Sign($close - Ref($close, 1)), 20)",
        "description": "20 日能量潮",
    },
    {
        "name": "obv_ratio",
        "display_name": "OBV比率",
        "category": "interaction",
        "expression": "Sum($volume * Sign($close - Ref($close, 1)), 5) / (Abs(Sum($volume * Sign($close - Ref($close, 1)), 20)) + 1e-8)",
        "description": "OBV 短長比",
    },
]

# 成交金額
AMOUNT_FACTORS = [
    {
        "name": "amount_5",
        "display_name": "成交額_5日",
        "category": "interaction",
        "expression": "Mean($close * $volume, 5)",
        "description": "5 日平均成交金額",
    },
    {
        "name": "amount_ratio",
        "display_name": "成交額比率",
        "category": "interaction",
        "expression": "Mean($close * $volume, 5) / (Mean($close * $volume, 20) + 1e-8)",
        "description": "成交額短長比",
    },
]

# 高低量
HIGH_LOW_VOL_FACTORS = [
    {
        "name": "high_vol_corr",
        "display_name": "高點量相關",
        "category": "interaction",
        "expression": "Corr($high, $volume, 20)",
        "description": "最高價與成交量 20 日相關",
    },
    {
        "name": "low_vol_corr",
        "display_name": "低點量相關",
        "category": "interaction",
        "expression": "Corr($low, $volume, 20)",
        "description": "最低價與成交量 20 日相關",
    },
]

# 振幅量
RANGE_VOL_FACTORS = [
    {
        "name": "range_vol_ratio",
        "display_name": "振幅量比",
        "category": "interaction",
        "expression": "(($high - $low) / $close) / ($volume / (Mean($volume, 20) + 1e-8) + 1e-8)",
        "description": "振幅 / 量能比",
    },
    {
        "name": "range_vol_corr",
        "display_name": "振幅量相關",
        "category": "interaction",
        "expression": "Corr(($high - $low) / $close, $volume, 20)",
        "description": "振幅與成交量相關",
    },
]


# =============================================================================
# 籌碼價格交互因子
# =============================================================================

# 外資與價格
FOREIGN_PRICE_FACTORS = [
    {
        "name": "foreign_price_corr",
        "display_name": "外資價格相關",
        "category": "interaction",
        "expression": "Corr($foreign_buy - $foreign_sell, $close, 20)",
        "description": "外資淨買與價格 20 日相關",
    },
    {
        "name": "foreign_ret_corr",
        "display_name": "外資收益相關",
        "category": "interaction",
        "expression": "Corr($foreign_buy - $foreign_sell, $close / Ref($close, 1) - 1, 20)",
        "description": "外資淨買與收益 20 日相關",
    },
    {
        "name": "foreign_lead_ret",
        "display_name": "外資領先收益",
        "category": "interaction",
        "expression": "Corr(Ref($foreign_buy - $foreign_sell, 1), $close / Ref($close, 1) - 1, 20)",
        "description": "外資領先 1 日與收益相關",
    },
]

# 投信與價格
TRUST_PRICE_FACTORS = [
    {
        "name": "trust_price_corr",
        "display_name": "投信價格相關",
        "category": "interaction",
        "expression": "Corr($trust_buy - $trust_sell, $close, 20)",
        "description": "投信淨買與價格 20 日相關",
    },
    {
        "name": "trust_ret_corr",
        "display_name": "投信收益相關",
        "category": "interaction",
        "expression": "Corr($trust_buy - $trust_sell, $close / Ref($close, 1) - 1, 20)",
        "description": "投信淨買與收益 20 日相關",
    },
]

# 法人與量
INST_VOL_CORR_FACTORS = [
    {
        "name": "foreign_vol_corr",
        "display_name": "外資量相關",
        "category": "interaction",
        "expression": "Corr($foreign_buy - $foreign_sell, $volume, 20)",
        "description": "外資淨買與成交量 20 日相關",
    },
    {
        "name": "trust_vol_corr",
        "display_name": "投信量相關",
        "category": "interaction",
        "expression": "Corr($trust_buy - $trust_sell, $volume, 20)",
        "description": "投信淨買與成交量 20 日相關",
    },
]

# 融資與價格
MARGIN_PRICE_FACTORS = [
    {
        "name": "margin_price_corr",
        "display_name": "融資價格相關",
        "category": "interaction",
        "expression": "Corr($margin_balance, $close, 20)",
        "description": "融資餘額與價格 20 日相關",
    },
    {
        "name": "margin_ret_corr",
        "display_name": "融資收益相關",
        "category": "interaction",
        "expression": "Corr($margin_balance - Ref($margin_balance, 1), $close / Ref($close, 1) - 1, 20)",
        "description": "融資變化與收益 20 日相關",
    },
]

# 融券與價格
SHORT_PRICE_FACTORS = [
    {
        "name": "short_price_corr",
        "display_name": "融券價格相關",
        "category": "interaction",
        "expression": "Corr($short_balance, $close, 20)",
        "description": "融券餘額與價格 20 日相關",
    },
    {
        "name": "short_ret_corr",
        "display_name": "融券收益相關",
        "category": "interaction",
        "expression": "Corr($short_balance - Ref($short_balance, 1), $close / Ref($close, 1) - 1, 20)",
        "description": "融券變化與收益 20 日相關",
    },
]

# 外資持股與價格
FOREIGN_RATIO_PRICE_FACTORS = [
    {
        "name": "fh_ratio_price_corr",
        "display_name": "持股價格相關",
        "category": "interaction",
        "expression": "Corr($foreign_ratio, $close, 60)",
        "description": "外資持股與價格 60 日相關",
    },
    {
        "name": "fh_ratio_ret_corr",
        "display_name": "持股收益相關",
        "category": "interaction",
        "expression": "Corr($foreign_ratio - Ref($foreign_ratio, 1), $close / Ref($close, 1) - 1, 20)",
        "description": "持股變化與收益 20 日相關",
    },
]

# 法人與波動
INST_VOLATILITY_FACTORS = [
    {
        "name": "foreign_volatility_corr",
        "display_name": "外資波動相關",
        "category": "interaction",
        "expression": "Corr(Abs($foreign_buy - $foreign_sell), Std($close / Ref($close, 1) - 1, 5), 20)",
        "description": "外資強度與波動相關",
    },
    {
        "name": "margin_volatility_corr",
        "display_name": "融資波動相關",
        "category": "interaction",
        "expression": "Corr(Abs($margin_balance - Ref($margin_balance, 1)), Std($close / Ref($close, 1) - 1, 5), 20)",
        "description": "融資變化與波動相關",
    },
]

# 法人順勢逆勢
INST_MOMENTUM_SYNC_FACTORS = [
    {
        "name": "foreign_momentum_sync",
        "display_name": "外資順勢程度",
        "category": "interaction",
        "expression": "Sum(Sign($foreign_buy - $foreign_sell) * Sign($close - Ref($close, 1)), 20) / 20",
        "description": "外資與價格同向比例",
    },
    {
        "name": "trust_momentum_sync",
        "display_name": "投信順勢程度",
        "category": "interaction",
        "expression": "Sum(Sign($trust_buy - $trust_sell) * Sign($close - Ref($close, 1)), 20) / 20",
        "description": "投信與價格同向比例",
    },
    {
        "name": "margin_contrarian_sync",
        "display_name": "散戶逆勢程度",
        "category": "interaction",
        "expression": "Sum(Sign($margin_balance - Ref($margin_balance, 1)) * Sign($close - Ref($close, 1)) * -1, 20) / 20",
        "description": "融資與價格反向比例",
    },
]

# 籌碼集中度與收益
INST_CONCENTRATION_FACTORS = [
    {
        "name": "inst_concentration_ret",
        "display_name": "法人集中收益",
        "category": "interaction",
        "expression": "(($foreign_buy + $trust_buy + $dealer_buy) - ($foreign_sell + $trust_sell + $dealer_sell)) / ($volume + 1e-8) * ($close / Ref($close, 1) - 1)",
        "description": "法人集中度與收益的乘積",
    },
]


# =============================================================================
# 估值動能交互因子
# =============================================================================

# 估值與動能（注意：pe_momentum 已在基礎因子中，這裡使用不同命名）
VALUATION_MOMENTUM_FACTORS = [
    {
        "name": "pe_momentum_ext",
        "display_name": "PE動能延伸",
        "category": "interaction",
        "expression": "$pe_ratio / (Mean($pe_ratio, 60) + 1e-8) - 1",
        "description": "PE 相對 60 日均值偏離",
    },
    {
        "name": "pb_momentum_ext",
        "display_name": "PB動能延伸",
        "category": "interaction",
        "expression": "$pb_ratio / (Mean($pb_ratio, 60) + 1e-8) - 1",
        "description": "PB 相對 60 日均值偏離",
    },
    {
        "name": "yield_momentum",
        "display_name": "殖利率動能",
        "category": "interaction",
        "expression": "$dividend_yield - Mean($dividend_yield, 60)",
        "description": "殖利率相對 60 日均值偏離",
    },
]

# 估值與收益
VALUATION_RET_FACTORS = [
    {
        "name": "pe_ret_corr",
        "display_name": "PE收益相關",
        "category": "interaction",
        "expression": "Corr($pe_ratio, $close / Ref($close, 1) - 1, 60)",
        "description": "PE 與收益 60 日相關",
    },
    {
        "name": "pb_ret_corr",
        "display_name": "PB收益相關",
        "category": "interaction",
        "expression": "Corr($pb_ratio, $close / Ref($close, 1) - 1, 60)",
        "description": "PB 與收益 60 日相關",
    },
]

# 估值變化
VALUATION_CHG_FACTORS = [
    {
        "name": "pe_chg_5d",
        "display_name": "PE變化_5日",
        "category": "interaction",
        "expression": "$pe_ratio - Ref($pe_ratio, 5)",
        "description": "PE 5 日變化",
    },
    {
        "name": "pe_chg_20d",
        "display_name": "PE變化_20日",
        "category": "interaction",
        "expression": "$pe_ratio - Ref($pe_ratio, 20)",
        "description": "PE 20 日變化",
    },
    {
        "name": "pb_chg_5d",
        "display_name": "PB變化_5日",
        "category": "interaction",
        "expression": "$pb_ratio - Ref($pb_ratio, 5)",
        "description": "PB 5 日變化",
    },
    {
        "name": "pb_chg_20d",
        "display_name": "PB變化_20日",
        "category": "interaction",
        "expression": "$pb_ratio - Ref($pb_ratio, 20)",
        "description": "PB 20 日變化",
    },
]

# 估值排名
VALUATION_RANK_FACTORS = [
    {
        "name": "pe_rank_60d",
        "display_name": "PE排名_60日",
        "category": "interaction",
        "expression": "Rank($pe_ratio, 60)",
        "description": "PE 在 60 日內的排名",
    },
    {
        "name": "pb_rank_60d",
        "display_name": "PB排名_60日",
        "category": "interaction",
        "expression": "Rank($pb_ratio, 60)",
        "description": "PB 在 60 日內的排名",
    },
]

# 價值動能組合
VALUE_MOMENTUM_COMBO_FACTORS = [
    {
        "name": "value_momentum_combo",
        "display_name": "價值動能組合",
        "category": "interaction",
        "expression": "Rank(1 / ($pe_ratio + 1e-8), 60) * ($close / Ref($close, 20) - 1)",
        "description": "價值排名乘以動能",
    },
    {
        "name": "growth_value_score",
        "display_name": "成長價值分數",
        "category": "interaction",
        "expression": "Rank($revenue / (Ref($revenue, 252) + 1e-8) - 1, 60) + Rank(1 / ($pe_ratio + 1e-8), 60)",
        "description": "成長排名 + 價值排名",
    },
]

# 估值與外資
VALUATION_FOREIGN_FACTORS = [
    {
        "name": "pe_foreign_sync",
        "display_name": "PE外資同向",
        "category": "interaction",
        "expression": "Corr($pe_ratio, $foreign_ratio, 60)",
        "description": "PE 與外資持股 60 日相關",
    },
]


# =============================================================================
# 匯出所有交互因子
# =============================================================================

INTERACTION_FACTORS = (
    # 價量交互（約 18 個）
    VOL_PRICE_CORR_FACTORS
    + RET_VOL_CORR_FACTORS
    + VOL_DISTRIBUTION_FACTORS
    + OBV_FACTORS
    + AMOUNT_FACTORS
    + HIGH_LOW_VOL_FACTORS
    + RANGE_VOL_FACTORS
    # 籌碼價格交互（約 22 個）
    + FOREIGN_PRICE_FACTORS
    + TRUST_PRICE_FACTORS
    + INST_VOL_CORR_FACTORS
    + MARGIN_PRICE_FACTORS
    + SHORT_PRICE_FACTORS
    + FOREIGN_RATIO_PRICE_FACTORS
    + INST_VOLATILITY_FACTORS
    + INST_MOMENTUM_SYNC_FACTORS
    + INST_CONCENTRATION_FACTORS
    # 估值動能交互（約 15 個）
    + VALUATION_MOMENTUM_FACTORS
    + VALUATION_RET_FACTORS
    + VALUATION_CHG_FACTORS
    + VALUATION_RANK_FACTORS
    + VALUE_MOMENTUM_COMBO_FACTORS
    + VALUATION_FOREIGN_FACTORS
)
