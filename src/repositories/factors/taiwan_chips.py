"""
台股籌碼因子

利用台股獨有的三大法人、融資融券、外資持股、借券、月營收等資料。
共計約 130 個因子。
"""

# =============================================================================
# 三大法人進階因子
# =============================================================================

# 外資累計淨買（多時間窗口）
FOREIGN_NET_FACTORS = [
    {
        "name": "foreign_net_10d",
        "display_name": "外資10日淨買",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 10)",
        "description": "外資 10 日累計淨買超",
    },
    {
        "name": "foreign_net_20d",
        "display_name": "外資20日淨買",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 20)",
        "description": "外資 20 日累計淨買超",
    },
    {
        "name": "foreign_net_60d",
        "display_name": "外資60日淨買",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 60)",
        "description": "外資 60 日累計淨買超",
    },
]

# 投信累計淨買
TRUST_NET_FACTORS = [
    {
        "name": "trust_net_10d",
        "display_name": "投信10日淨買",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 10)",
        "description": "投信 10 日累計淨買超",
    },
    {
        "name": "trust_net_20d",
        "display_name": "投信20日淨買",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 20)",
        "description": "投信 20 日累計淨買超",
    },
    {
        "name": "trust_net_60d",
        "display_name": "投信60日淨買",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 60)",
        "description": "投信 60 日累計淨買超",
    },
]

# 自營商累計淨買
DEALER_NET_FACTORS = [
    {
        "name": "dealer_net_5d",
        "display_name": "自營商5日淨買",
        "category": "chips",
        "expression": "Sum($dealer_buy - $dealer_sell, 5)",
        "description": "自營商 5 日累計淨買超",
    },
    {
        "name": "dealer_net_10d",
        "display_name": "自營商10日淨買",
        "category": "chips",
        "expression": "Sum($dealer_buy - $dealer_sell, 10)",
        "description": "自營商 10 日累計淨買超",
    },
    {
        "name": "dealer_net_20d",
        "display_name": "自營商20日淨買",
        "category": "chips",
        "expression": "Sum($dealer_buy - $dealer_sell, 20)",
        "description": "自營商 20 日累計淨買超",
    },
]

# 三大法人合計
INST_NET_FACTORS = [
    {
        "name": "inst_net",
        "display_name": "三大法人淨買",
        "category": "chips",
        "expression": "($foreign_buy - $foreign_sell) + ($trust_buy - $trust_sell) + ($dealer_buy - $dealer_sell)",
        "description": "三大法人當日淨買超合計",
    },
    {
        "name": "inst_net_5d",
        "display_name": "三大法人5日淨買",
        "category": "chips",
        "expression": "Sum(($foreign_buy - $foreign_sell) + ($trust_buy - $trust_sell) + ($dealer_buy - $dealer_sell), 5)",
        "description": "三大法人 5 日累計淨買超",
    },
    {
        "name": "inst_net_10d",
        "display_name": "三大法人10日淨買",
        "category": "chips",
        "expression": "Sum(($foreign_buy - $foreign_sell) + ($trust_buy - $trust_sell) + ($dealer_buy - $dealer_sell), 10)",
        "description": "三大法人 10 日累計淨買超",
    },
    {
        "name": "inst_net_20d",
        "display_name": "三大法人20日淨買",
        "category": "chips",
        "expression": "Sum(($foreign_buy - $foreign_sell) + ($trust_buy - $trust_sell) + ($dealer_buy - $dealer_sell), 20)",
        "description": "三大法人 20 日累計淨買超",
    },
]

# 法人動能變化率
INST_MOMENTUM_FACTORS = [
    {
        "name": "foreign_net_mom",
        "display_name": "外資淨買動能",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 5) / (Abs(Sum($foreign_buy - $foreign_sell, 20)) + 1e-8)",
        "description": "外資 5 日淨買 / |20 日淨買|",
    },
    {
        "name": "trust_net_mom",
        "display_name": "投信淨買動能",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 5) / (Abs(Sum($trust_buy - $trust_sell, 20)) + 1e-8)",
        "description": "投信 5 日淨買 / |20 日淨買|",
    },
    {
        "name": "dealer_net_mom",
        "display_name": "自營商淨買動能",
        "category": "chips",
        "expression": "Sum($dealer_buy - $dealer_sell, 5) / (Abs(Sum($dealer_buy - $dealer_sell, 20)) + 1e-8)",
        "description": "自營商 5 日淨買 / |20 日淨買|",
    },
]

# 法人買賣強度
INST_INTENSITY_FACTORS = [
    {
        "name": "foreign_buy_intensity",
        "display_name": "外資買入強度",
        "category": "chips",
        "expression": "$foreign_buy / (Mean($foreign_buy, 20) + 1e-8)",
        "description": "外資當日買入 / 20 日均買入",
    },
    {
        "name": "foreign_sell_intensity",
        "display_name": "外資賣出強度",
        "category": "chips",
        "expression": "$foreign_sell / (Mean($foreign_sell, 20) + 1e-8)",
        "description": "外資當日賣出 / 20 日均賣出",
    },
    {
        "name": "trust_buy_intensity",
        "display_name": "投信買入強度",
        "category": "chips",
        "expression": "$trust_buy / (Mean($trust_buy, 20) + 1e-8)",
        "description": "投信當日買入 / 20 日均買入",
    },
    {
        "name": "trust_sell_intensity",
        "display_name": "投信賣出強度",
        "category": "chips",
        "expression": "$trust_sell / (Mean($trust_sell, 20) + 1e-8)",
        "description": "投信當日賣出 / 20 日均賣出",
    },
]

# 法人連續買賣
INST_STREAK_FACTORS = [
    {
        "name": "foreign_buy_streak",
        "display_name": "外資連買天數",
        "category": "chips",
        "expression": "Sum(Greater($foreign_buy - $foreign_sell, 0), 10)",
        "description": "外資 10 日內淨買天數",
    },
    {
        "name": "trust_buy_streak",
        "display_name": "投信連買天數",
        "category": "chips",
        "expression": "Sum(Greater($trust_buy - $trust_sell, 0), 10)",
        "description": "投信 10 日內淨買天數",
    },
    {
        "name": "dealer_buy_streak",
        "display_name": "自營商連買天數",
        "category": "chips",
        "expression": "Sum(Greater($dealer_buy - $dealer_sell, 0), 10)",
        "description": "自營商 10 日內淨買天數",
    },
]

# 法人同向性
INST_AGREEMENT_FACTORS = [
    {
        "name": "inst_agreement",
        "display_name": "法人同向性",
        "category": "chips",
        "expression": "Sign($foreign_buy - $foreign_sell) + Sign($trust_buy - $trust_sell) + Sign($dealer_buy - $dealer_sell)",
        "description": "三大法人同向程度 (-3 到 3)",
    },
    {
        "name": "inst_agreement_5d",
        "display_name": "法人5日同向性",
        "category": "chips",
        "expression": "Sum(Sign($foreign_buy - $foreign_sell) + Sign($trust_buy - $trust_sell) + Sign($dealer_buy - $dealer_sell), 5)",
        "description": "三大法人 5 日同向程度",
    },
    {
        "name": "foreign_trust_sync",
        "display_name": "外投同向",
        "category": "chips",
        "expression": "Sign($foreign_buy - $foreign_sell) * Sign($trust_buy - $trust_sell)",
        "description": "外資與投信是否同向",
    },
    {
        "name": "foreign_trust_sync_5d",
        "display_name": "外投5日同向",
        "category": "chips",
        "expression": "Sum(Sign($foreign_buy - $foreign_sell) * Sign($trust_buy - $trust_sell), 5)",
        "description": "外資與投信 5 日同向次數",
    },
]

# 法人買賣比
INST_BS_RATIO_FACTORS = [
    {
        "name": "foreign_bs_ratio",
        "display_name": "外資買賣比",
        "category": "chips",
        "expression": "$foreign_buy / ($foreign_sell + 1)",
        "description": "外資買入 / 賣出比",
    },
    {
        "name": "trust_bs_ratio",
        "display_name": "投信買賣比",
        "category": "chips",
        "expression": "$trust_buy / ($trust_sell + 1)",
        "description": "投信買入 / 賣出比",
    },
    {
        "name": "dealer_bs_ratio",
        "display_name": "自營商買賣比",
        "category": "chips",
        "expression": "$dealer_buy / ($dealer_sell + 1)",
        "description": "自營商買入 / 賣出比",
    },
]

# 法人佔成交量比
INST_VOL_RATIO_FACTORS = [
    {
        "name": "foreign_vol_ratio",
        "display_name": "外資佔成交量",
        "category": "chips",
        "expression": "($foreign_buy + $foreign_sell) / ($volume + 1e-8)",
        "description": "外資買賣佔成交量比",
    },
    {
        "name": "trust_vol_ratio",
        "display_name": "投信佔成交量",
        "category": "chips",
        "expression": "($trust_buy + $trust_sell) / ($volume + 1e-8)",
        "description": "投信買賣佔成交量比",
    },
    {
        "name": "dealer_vol_ratio",
        "display_name": "自營商佔成交量",
        "category": "chips",
        "expression": "($dealer_buy + $dealer_sell) / ($volume + 1e-8)",
        "description": "自營商買賣佔成交量比",
    },
    {
        "name": "inst_vol_ratio",
        "display_name": "法人佔成交量",
        "category": "chips",
        "expression": "(($foreign_buy + $foreign_sell) + ($trust_buy + $trust_sell) + ($dealer_buy + $dealer_sell)) / ($volume + 1e-8)",
        "description": "三大法人佔成交量比",
    },
]

# 法人淨買佔成交量
INST_NET_VOL_FACTORS = [
    {
        "name": "foreign_net_vol",
        "display_name": "外資淨買佔比",
        "category": "chips",
        "expression": "($foreign_buy - $foreign_sell) / ($volume + 1e-8)",
        "description": "外資淨買佔成交量比",
    },
    {
        "name": "trust_net_vol",
        "display_name": "投信淨買佔比",
        "category": "chips",
        "expression": "($trust_buy - $trust_sell) / ($volume + 1e-8)",
        "description": "投信淨買佔成交量比",
    },
    {
        "name": "inst_net_vol",
        "display_name": "法人淨買佔比",
        "category": "chips",
        "expression": "(($foreign_buy - $foreign_sell) + ($trust_buy - $trust_sell) + ($dealer_buy - $dealer_sell)) / ($volume + 1e-8)",
        "description": "三大法人淨買佔成交量比",
    },
    {
        "name": "foreign_net_vol_5d",
        "display_name": "外資5日淨買佔比",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 5) / (Sum($volume, 5) + 1e-8)",
        "description": "外資 5 日淨買佔 5 日成交量比",
    },
    {
        "name": "foreign_net_vol_20d",
        "display_name": "外資20日淨買佔比",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 20) / (Sum($volume, 20) + 1e-8)",
        "description": "外資 20 日淨買佔 20 日成交量比",
    },
    {
        "name": "trust_net_vol_5d",
        "display_name": "投信5日淨買佔比",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 5) / (Sum($volume, 5) + 1e-8)",
        "description": "投信 5 日淨買佔 5 日成交量比",
    },
    {
        "name": "trust_net_vol_20d",
        "display_name": "投信20日淨買佔比",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 20) / (Sum($volume, 20) + 1e-8)",
        "description": "投信 20 日淨買佔 20 日成交量比",
    },
]


# =============================================================================
# 融資融券進階因子
# =============================================================================

# 融資餘額變化
MARGIN_CHG_FACTORS = [
    {
        "name": "margin_chg",
        "display_name": "融資餘額日變化",
        "category": "chips",
        "expression": "$margin_balance - Ref($margin_balance, 1)",
        "description": "融資餘額日變化量",
    },
    {
        "name": "margin_chg_10d",
        "display_name": "融資餘額10日變化",
        "category": "chips",
        "expression": "$margin_balance - Ref($margin_balance, 10)",
        "description": "融資餘額 10 日變化量",
    },
    {
        "name": "margin_chg_20d",
        "display_name": "融資餘額20日變化",
        "category": "chips",
        "expression": "$margin_balance - Ref($margin_balance, 20)",
        "description": "融資餘額 20 日變化量",
    },
    {
        "name": "margin_chg_pct",
        "display_name": "融資餘額變化率",
        "category": "chips",
        "expression": "$margin_balance / (Ref($margin_balance, 5) + 1e-8) - 1",
        "description": "融資餘額 5 日變化率",
    },
]

# 融券餘額變化
SHORT_CHG_FACTORS = [
    {
        "name": "short_chg",
        "display_name": "融券餘額日變化",
        "category": "chips",
        "expression": "$short_balance - Ref($short_balance, 1)",
        "description": "融券餘額日變化量",
    },
    {
        "name": "short_chg_10d",
        "display_name": "融券餘額10日變化",
        "category": "chips",
        "expression": "$short_balance - Ref($short_balance, 10)",
        "description": "融券餘額 10 日變化量",
    },
    {
        "name": "short_chg_20d",
        "display_name": "融券餘額20日變化",
        "category": "chips",
        "expression": "$short_balance - Ref($short_balance, 20)",
        "description": "融券餘額 20 日變化量",
    },
    {
        "name": "short_chg_pct",
        "display_name": "融券餘額變化率",
        "category": "chips",
        "expression": "$short_balance / (Ref($short_balance, 5) + 1e-8) - 1",
        "description": "融券餘額 5 日變化率",
    },
]

# 券資比
SHORT_MARGIN_FACTORS = [
    {
        "name": "short_margin_ratio",
        "display_name": "券資比",
        "category": "chips",
        "expression": "$short_balance / ($margin_balance + 1)",
        "description": "融券餘額 / 融資餘額",
    },
    {
        "name": "short_margin_ratio_chg",
        "display_name": "券資比變化",
        "category": "chips",
        "expression": "$short_balance / ($margin_balance + 1) - Ref($short_balance, 5) / (Ref($margin_balance, 5) + 1)",
        "description": "券資比 5 日變化",
    },
]

# 融資融券淨買賣
MARGIN_NET_FACTORS = [
    {
        "name": "margin_net",
        "display_name": "融資淨買",
        "category": "chips",
        "expression": "$margin_buy - $margin_sell",
        "description": "融資淨買入量",
    },
    {
        "name": "margin_net_5d",
        "display_name": "融資5日淨買",
        "category": "chips",
        "expression": "Sum($margin_buy - $margin_sell, 5)",
        "description": "融資 5 日累計淨買入",
    },
    {
        "name": "margin_net_20d",
        "display_name": "融資20日淨買",
        "category": "chips",
        "expression": "Sum($margin_buy - $margin_sell, 20)",
        "description": "融資 20 日累計淨買入",
    },
    {
        "name": "short_net",
        "display_name": "融券淨賣",
        "category": "chips",
        "expression": "$short_sell - $short_buy",
        "description": "融券淨賣出量",
    },
    {
        "name": "short_net_5d",
        "display_name": "融券5日淨賣",
        "category": "chips",
        "expression": "Sum($short_sell - $short_buy, 5)",
        "description": "融券 5 日累計淨賣出",
    },
    {
        "name": "short_net_20d",
        "display_name": "融券20日淨賣",
        "category": "chips",
        "expression": "Sum($short_sell - $short_buy, 20)",
        "description": "融券 20 日累計淨賣出",
    },
]

# 融資融券動能
MARGIN_MOM_FACTORS = [
    {
        "name": "margin_mom_5_20",
        "display_name": "融資動能",
        "category": "chips",
        "expression": "Mean($margin_balance, 5) / (Mean($margin_balance, 20) + 1e-8)",
        "description": "融資 5 日均值 / 20 日均值",
    },
    {
        "name": "short_mom_5_20",
        "display_name": "融券動能",
        "category": "chips",
        "expression": "Mean($short_balance, 5) / (Mean($short_balance, 20) + 1e-8)",
        "description": "融券 5 日均值 / 20 日均值",
    },
]

# 融資融券強度
MARGIN_INTENSITY_FACTORS = [
    {
        "name": "margin_intensity",
        "display_name": "融資使用強度",
        "category": "chips",
        "expression": "$margin_buy / (Mean($margin_buy, 20) + 1e-8)",
        "description": "融資買入 / 20 日均值",
    },
    {
        "name": "short_intensity",
        "display_name": "融券使用強度",
        "category": "chips",
        "expression": "$short_sell / (Mean($short_sell, 20) + 1e-8)",
        "description": "融券賣出 / 20 日均值",
    },
]

# 融資融券買賣比
MARGIN_BS_RATIO_FACTORS = [
    {
        "name": "margin_bs_ratio",
        "display_name": "融資買賣比",
        "category": "chips",
        "expression": "$margin_buy / ($margin_sell + 1)",
        "description": "融資買入 / 賣出比",
    },
    {
        "name": "short_bs_ratio",
        "display_name": "融券買賣比",
        "category": "chips",
        "expression": "$short_buy / ($short_sell + 1)",
        "description": "融券買入 / 賣出比（回補 / 放空）",
    },
]

# 融資融券佔成交量
MARGIN_VOL_FACTORS = [
    {
        "name": "margin_vol_ratio",
        "display_name": "融資佔成交量",
        "category": "chips",
        "expression": "($margin_buy + $margin_sell) / ($volume + 1e-8)",
        "description": "融資買賣佔成交量比",
    },
    {
        "name": "short_vol_ratio",
        "display_name": "融券佔成交量",
        "category": "chips",
        "expression": "($short_buy + $short_sell) / ($volume + 1e-8)",
        "description": "融券買賣佔成交量比",
    },
]

# 融資融券排名
MARGIN_RANK_FACTORS = [
    {
        "name": "margin_rank_20d",
        "display_name": "融資變化排名",
        "category": "chips",
        "expression": "Rank($margin_balance - Ref($margin_balance, 1), 20)",
        "description": "融資餘額日變化 20 日排名",
    },
    {
        "name": "short_rank_20d",
        "display_name": "融券變化排名",
        "category": "chips",
        "expression": "Rank($short_balance - Ref($short_balance, 1), 20)",
        "description": "融券餘額日變化 20 日排名",
    },
]

# 融資融券波動
MARGIN_STD_FACTORS = [
    {
        "name": "margin_std_20d",
        "display_name": "融資波動",
        "category": "chips",
        "expression": "Std($margin_balance, 20) / (Mean($margin_balance, 20) + 1e-8)",
        "description": "融資餘額 20 日變異係數",
    },
    {
        "name": "short_std_20d",
        "display_name": "融券波動",
        "category": "chips",
        "expression": "Std($short_balance, 20) / (Mean($short_balance, 20) + 1e-8)",
        "description": "融券餘額 20 日變異係數",
    },
]

# 散戶逆向指標
RETAIL_CONTRARIAN_FACTORS = [
    {
        "name": "retail_contrarian",
        "display_name": "散戶逆向",
        "category": "chips",
        "expression": "($margin_balance - Ref($margin_balance, 5)) / (Ref($margin_balance, 5) + 1e-8) * -1",
        "description": "融資增減反向指標",
    },
]


# =============================================================================
# 外資持股進階因子
# =============================================================================

# 外資持股變化
FOREIGN_RATIO_CHG_FACTORS = [
    {
        "name": "foreign_ratio_chg_1d",
        "display_name": "外資持股日變化",
        "category": "chips",
        "expression": "$foreign_ratio - Ref($foreign_ratio, 1)",
        "description": "外資持股比率日變化",
    },
    {
        "name": "foreign_ratio_chg_10d",
        "display_name": "外資持股10日變化",
        "category": "chips",
        "expression": "$foreign_ratio - Ref($foreign_ratio, 10)",
        "description": "外資持股比率 10 日變化",
    },
    {
        "name": "foreign_ratio_chg_20d",
        "display_name": "外資持股20日變化",
        "category": "chips",
        "expression": "$foreign_ratio - Ref($foreign_ratio, 20)",
        "description": "外資持股比率 20 日變化",
    },
    {
        "name": "foreign_ratio_chg_60d",
        "display_name": "外資持股60日變化",
        "category": "chips",
        "expression": "$foreign_ratio - Ref($foreign_ratio, 60)",
        "description": "外資持股比率 60 日變化",
    },
]

# 外資持股動能
FOREIGN_RATIO_MOM_FACTORS = [
    {
        "name": "foreign_ratio_mom",
        "display_name": "外資持股動能",
        "category": "chips",
        "expression": "Mean($foreign_ratio, 5) / (Mean($foreign_ratio, 20) + 1e-8)",
        "description": "外資持股 5 日均 / 20 日均",
    },
    {
        "name": "foreign_ratio_trend",
        "display_name": "外資持股趨勢",
        "category": "chips",
        "expression": "Slope($foreign_ratio, 20)",
        "description": "外資持股比率 20 日斜率",
    },
]

# 外資持股排名
FOREIGN_RATIO_RANK_FACTORS = [
    {
        "name": "foreign_ratio_rank",
        "display_name": "外資持股排名",
        "category": "chips",
        "expression": "Rank($foreign_ratio, 60)",
        "description": "外資持股比率 60 日排名",
    },
    {
        "name": "foreign_chg_rank",
        "display_name": "外資變化排名",
        "category": "chips",
        "expression": "Rank($foreign_ratio - Ref($foreign_ratio, 5), 20)",
        "description": "外資持股變化 20 日排名",
    },
]

# 外資可投資空間
FOREIGN_SPACE_FACTORS = [
    {
        "name": "foreign_space",
        "display_name": "外資可投資空間",
        "category": "chips",
        "expression": "$foreign_remaining_ratio",
        "description": "外資尚可投資比率",
    },
    {
        "name": "foreign_space_used",
        "display_name": "外資額度使用率",
        "category": "chips",
        "expression": "$foreign_ratio / ($foreign_upper_limit_ratio + 1e-8)",
        "description": "外資持股 / 投資上限",
    },
    {
        "name": "foreign_space_chg",
        "display_name": "外資空間變化",
        "category": "chips",
        "expression": "$foreign_remaining_ratio - Ref($foreign_remaining_ratio, 5)",
        "description": "外資可投資空間 5 日變化",
    },
]

# 外資持股一致性
FOREIGN_CONSISTENCY_FACTORS = [
    {
        "name": "foreign_consistency",
        "display_name": "外資買賣一致性",
        "category": "chips",
        "expression": "Sign($foreign_ratio - Ref($foreign_ratio, 1)) * Sign($foreign_buy - $foreign_sell)",
        "description": "外資持股變化與買賣方向一致性",
    },
    {
        "name": "foreign_consistency_5d",
        "display_name": "外資5日一致性",
        "category": "chips",
        "expression": "Sum(Sign($foreign_ratio - Ref($foreign_ratio, 1)) * Sign($foreign_buy - $foreign_sell), 5)",
        "description": "外資 5 日買賣一致性累計",
    },
]

# 外資持股穩定性
FOREIGN_STABILITY_FACTORS = [
    {
        "name": "foreign_ratio_std",
        "display_name": "外資持股波動",
        "category": "chips",
        "expression": "Std($foreign_ratio, 20)",
        "description": "外資持股比率 20 日標準差",
    },
    {
        "name": "foreign_ratio_cv",
        "display_name": "外資持股變異",
        "category": "chips",
        "expression": "Std($foreign_ratio, 20) / (Mean($foreign_ratio, 20) + 1e-8)",
        "description": "外資持股變異係數",
    },
]

# 外資持股加速度
FOREIGN_ACCEL_FACTORS = [
    {
        "name": "foreign_accel",
        "display_name": "外資持股加速度",
        "category": "chips",
        "expression": "($foreign_ratio - Ref($foreign_ratio, 5)) - (Ref($foreign_ratio, 5) - Ref($foreign_ratio, 10))",
        "description": "外資持股變化加速度",
    },
]


# =============================================================================
# 借券進階因子
# =============================================================================

LENDING_FACTORS = [
    {
        "name": "lending_chg",
        "display_name": "借券量日變化",
        "category": "chips",
        "expression": "$lending_volume - Ref($lending_volume, 1)",
        "description": "借券量日變化",
    },
    {
        "name": "lending_chg_5d",
        "display_name": "借券量5日變化",
        "category": "chips",
        "expression": "$lending_volume - Ref($lending_volume, 5)",
        "description": "借券量 5 日變化",
    },
    {
        "name": "lending_chg_20d",
        "display_name": "借券量20日變化",
        "category": "chips",
        "expression": "$lending_volume - Ref($lending_volume, 20)",
        "description": "借券量 20 日變化",
    },
    {
        "name": "lending_ratio",
        "display_name": "借券佔成交量",
        "category": "chips",
        "expression": "$lending_volume / ($volume + 1e-8)",
        "description": "借券量 / 成交量",
    },
    {
        "name": "lending_ratio_5d",
        "display_name": "借券5日佔比",
        "category": "chips",
        "expression": "Sum($lending_volume, 5) / (Sum($volume, 5) + 1e-8)",
        "description": "5 日借券量 / 5 日成交量",
    },
    {
        "name": "lending_ratio_20d",
        "display_name": "借券20日佔比",
        "category": "chips",
        "expression": "Sum($lending_volume, 20) / (Sum($volume, 20) + 1e-8)",
        "description": "20 日借券量 / 20 日成交量",
    },
    {
        "name": "lending_mom",
        "display_name": "借券動能",
        "category": "chips",
        "expression": "Mean($lending_volume, 5) / (Mean($lending_volume, 20) + 1e-8)",
        "description": "借券 5 日均 / 20 日均",
    },
    {
        "name": "lending_intensity_chg",
        "display_name": "借券強度變化",
        "category": "chips",
        "expression": "$lending_volume / ($volume + 1e-8) - Ref($lending_volume, 5) / (Ref($volume, 5) + 1e-8)",
        "description": "借券強度 5 日變化",
    },
    {
        "name": "lending_rank",
        "display_name": "借券量排名",
        "category": "chips",
        "expression": "Rank($lending_volume, 60)",
        "description": "借券量 60 日排名",
    },
    {
        "name": "lending_chg_rank",
        "display_name": "借券變化排名",
        "category": "chips",
        "expression": "Rank($lending_volume - Ref($lending_volume, 1), 20)",
        "description": "借券變化 20 日排名",
    },
    {
        "name": "lending_std",
        "display_name": "借券波動",
        "category": "chips",
        "expression": "Std($lending_volume, 20) / (Mean($lending_volume, 20) + 1e-8)",
        "description": "借券量 20 日變異係數",
    },
]


# =============================================================================
# 月營收進階因子
# =============================================================================

REVENUE_ADV_FACTORS = [
    {
        "name": "revenue_mom_qoq",
        "display_name": "營收季增",
        "category": "revenue",
        "expression": "$revenue / (Ref($revenue, 63) + 1e-8) - 1",
        "description": "月營收季增率（約 63 交易日）",
    },
    {
        "name": "revenue_growth_acc",
        "display_name": "營收成長加速",
        "category": "revenue",
        "expression": "$revenue / (Ref($revenue, 21) + 1e-8) - Ref($revenue, 21) / (Ref($revenue, 42) + 1e-8)",
        "description": "營收月增率加速度",
    },
    {
        "name": "revenue_momentum_3m",
        "display_name": "營收3月動能",
        "category": "revenue",
        "expression": "Mean($revenue, 63) / (Mean($revenue, 252) + 1e-8)",
        "description": "3 個月營收 / 12 個月營收",
    },
    {
        "name": "revenue_momentum_6m",
        "display_name": "營收6月動能",
        "category": "revenue",
        "expression": "Mean($revenue, 126) / (Mean($revenue, 252) + 1e-8)",
        "description": "6 個月營收 / 12 個月營收",
    },
    {
        "name": "revenue_trend",
        "display_name": "營收趨勢",
        "category": "revenue",
        "expression": "Slope($revenue, 63)",
        "description": "營收 3 個月斜率",
    },
    {
        "name": "revenue_trend_chg",
        "display_name": "營收趨勢變化",
        "category": "revenue",
        "expression": "Slope($revenue, 63) - Slope(Ref($revenue, 21), 63)",
        "description": "營收趨勢月變化",
    },
    {
        "name": "revenue_vol",
        "display_name": "營收波動率",
        "category": "revenue",
        "expression": "Std($revenue, 126) / (Mean($revenue, 126) + 1e-8)",
        "description": "營收 6 個月變異係數",
    },
    {
        "name": "revenue_yoy_rank",
        "display_name": "營收年增排名",
        "category": "revenue",
        "expression": "Rank($revenue / (Ref($revenue, 252) + 1e-8) - 1, 60)",
        "description": "營收年增率 60 日排名",
    },
    {
        "name": "revenue_surprise",
        "display_name": "營收驚喜",
        "category": "revenue",
        "expression": "($revenue - Mean($revenue, 63)) / (Std($revenue, 63) + 1e-8)",
        "description": "營收相對 3 個月均值的偏離",
    },
    {
        "name": "revenue_price_ratio",
        "display_name": "營收價格比",
        "category": "revenue",
        "expression": "$revenue / ($close * $total_shares + 1e-8)",
        "description": "月營收 / 市值",
    },
]


# =============================================================================
# 匯出所有台股籌碼因子
# =============================================================================

TAIWAN_CHIPS_FACTORS = (
    # 三大法人進階（約 50 個）
    FOREIGN_NET_FACTORS
    + TRUST_NET_FACTORS
    + DEALER_NET_FACTORS
    + INST_NET_FACTORS
    + INST_MOMENTUM_FACTORS
    + INST_INTENSITY_FACTORS
    + INST_STREAK_FACTORS
    + INST_AGREEMENT_FACTORS
    + INST_BS_RATIO_FACTORS
    + INST_VOL_RATIO_FACTORS
    + INST_NET_VOL_FACTORS
    # 融資融券進階（約 35 個）
    + MARGIN_CHG_FACTORS
    + SHORT_CHG_FACTORS
    + SHORT_MARGIN_FACTORS
    + MARGIN_NET_FACTORS
    + MARGIN_MOM_FACTORS
    + MARGIN_INTENSITY_FACTORS
    + MARGIN_BS_RATIO_FACTORS
    + MARGIN_VOL_FACTORS
    + MARGIN_RANK_FACTORS
    + MARGIN_STD_FACTORS
    + RETAIL_CONTRARIAN_FACTORS
    # 外資持股進階（約 25 個）
    + FOREIGN_RATIO_CHG_FACTORS
    + FOREIGN_RATIO_MOM_FACTORS
    + FOREIGN_RATIO_RANK_FACTORS
    + FOREIGN_SPACE_FACTORS
    + FOREIGN_CONSISTENCY_FACTORS
    + FOREIGN_STABILITY_FACTORS
    + FOREIGN_ACCEL_FACTORS
    # 借券進階（約 12 個）
    + LENDING_FACTORS
    # 月營收進階（約 10 個）
    + REVENUE_ADV_FACTORS
)
