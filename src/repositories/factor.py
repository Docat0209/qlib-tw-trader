from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.repositories.models import Factor, TrainingFactorResult

# 預設因子清單（30 個）
DEFAULT_FACTORS = [
    # 技術面（Technical）- 10 個
    {
        "name": "mom_5d",
        "display_name": "5日動能",
        "category": "technical",
        "expression": "$close / Ref($close, 5) - 1",
        "description": "5 日價格動能",
    },
    {
        "name": "mom_20d",
        "display_name": "20日動能",
        "category": "technical",
        "expression": "$close / Ref($close, 20) - 1",
        "description": "20 日價格動能",
    },
    {
        "name": "ma_ratio_5_20",
        "display_name": "均線比 5/20",
        "category": "technical",
        "expression": "Mean($close, 5) / Mean($close, 20)",
        "description": "5 日均線 / 20 日均線",
    },
    {
        "name": "volatility_20d",
        "display_name": "20日波動率",
        "category": "technical",
        "expression": "Std($close / Ref($close, 1) - 1, 20)",
        "description": "20 日收益率標準差",
    },
    {
        "name": "volume_ratio_5d",
        "display_name": "量比 5日",
        "category": "technical",
        "expression": "$volume / Mean($volume, 5)",
        "description": "當日成交量 / 5 日平均成交量",
    },
    {
        "name": "high_low_range",
        "display_name": "振幅",
        "category": "technical",
        "expression": "($high - $low) / $close",
        "description": "日內振幅比",
    },
    {
        "name": "close_position",
        "display_name": "K棒位置",
        "category": "technical",
        "expression": "($close - $low) / ($high - $low + 1e-8)",
        "description": "收盤價在日內區間的位置",
    },
    {
        "name": "turnover_change",
        "display_name": "換手率變化",
        "category": "technical",
        "expression": "Mean($volume, 20) / Mean($volume, 60)",
        "description": "短期換手率 / 長期換手率",
    },
    {
        "name": "price_rank_60d",
        "display_name": "動能排名",
        "category": "technical",
        "expression": "Rank($close / Ref($close, 20) - 1, 60)",
        "description": "20 日動能在 60 日內的排名",
    },
    {
        "name": "rsi_proxy",
        "display_name": "RSI 近似",
        "category": "technical",
        "expression": "Mean(Greater($close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 1e-8)",
        "description": "RSI 近似值",
    },
    # 籌碼面（Chips）- 10 個
    {
        "name": "foreign_net",
        "display_name": "外資淨買",
        "category": "chips",
        "expression": "$foreign_buy - $foreign_sell",
        "description": "外資當日淨買超張數",
    },
    {
        "name": "foreign_net_5d",
        "display_name": "外資 5日淨買",
        "category": "chips",
        "expression": "Sum($foreign_buy - $foreign_sell, 5)",
        "description": "外資 5 日累計淨買超",
    },
    {
        "name": "trust_net",
        "display_name": "投信淨買",
        "category": "chips",
        "expression": "$trust_buy - $trust_sell",
        "description": "投信當日淨買超張數",
    },
    {
        "name": "trust_net_5d",
        "display_name": "投信 5日淨買",
        "category": "chips",
        "expression": "Sum($trust_buy - $trust_sell, 5)",
        "description": "投信 5 日累計淨買超",
    },
    {
        "name": "dealer_net",
        "display_name": "自營商淨買",
        "category": "chips",
        "expression": "$dealer_buy - $dealer_sell",
        "description": "自營商當日淨買超張數",
    },
    {
        "name": "foreign_ratio_chg",
        "display_name": "外資持股變化",
        "category": "chips",
        "expression": "$foreign_ratio - Ref($foreign_ratio, 5)",
        "description": "外資持股比率 5 日變化",
    },
    {
        "name": "margin_balance_chg",
        "display_name": "融資餘額變化",
        "category": "chips",
        "expression": "$margin_balance - Ref($margin_balance, 5)",
        "description": "融資餘額 5 日變化",
    },
    {
        "name": "short_balance_chg",
        "display_name": "融券餘額變化",
        "category": "chips",
        "expression": "$short_balance - Ref($short_balance, 5)",
        "description": "融券餘額 5 日變化",
    },
    {
        "name": "margin_short_ratio",
        "display_name": "券資比",
        "category": "chips",
        "expression": "$margin_balance / ($short_balance + 1)",
        "description": "融資餘額 / 融券餘額",
    },
    {
        "name": "lending_intensity",
        "display_name": "借券強度",
        "category": "chips",
        "expression": "$lending_volume / ($volume + 1)",
        "description": "借券量 / 成交量",
    },
    # 估值面（Valuation）- 5 個
    {
        "name": "ep",
        "display_name": "益本比",
        "category": "valuation",
        "expression": "1 / ($pe_ratio + 1e-8)",
        "description": "本益比倒數（Earnings/Price）",
    },
    {
        "name": "bp",
        "display_name": "淨值股價比",
        "category": "valuation",
        "expression": "1 / ($pb_ratio + 1e-8)",
        "description": "股價淨值比倒數（Book/Price）",
    },
    {
        "name": "div_yield",
        "display_name": "殖利率",
        "category": "valuation",
        "expression": "$dividend_yield",
        "description": "股利殖利率",
    },
    {
        "name": "pe_momentum",
        "display_name": "本益比動能",
        "category": "valuation",
        "expression": "$pe_ratio / Mean($pe_ratio, 60) - 1",
        "description": "本益比相對 60 日均值的變化",
    },
    {
        "name": "value_score",
        "display_name": "價值分數",
        "category": "valuation",
        "expression": "Rank(1/($pe_ratio+1e-8), 60) + Rank(1/($pb_ratio+1e-8), 60)",
        "description": "益本比排名 + 淨值股價比排名",
    },
    # 營收面（Revenue）- 5 個
    {
        "name": "revenue_mom",
        "display_name": "營收月增",
        "category": "revenue",
        "expression": "$revenue / Ref($revenue, 21) - 1",
        "description": "月營收月增率（約 21 個交易日）",
    },
    {
        "name": "revenue_yoy",
        "display_name": "營收年增",
        "category": "revenue",
        "expression": "$revenue / Ref($revenue, 252) - 1",
        "description": "月營收年增率（約 252 個交易日）",
    },
    {
        "name": "revenue_ma_ratio",
        "display_name": "營收均線比",
        "category": "revenue",
        "expression": "Mean($revenue, 63) / Mean($revenue, 126) - 1",
        "description": "3 個月均營收 / 6 個月均營收",
    },
    {
        "name": "revenue_rank",
        "display_name": "營收排名",
        "category": "revenue",
        "expression": "Rank($revenue / Ref($revenue, 21) - 1, 60)",
        "description": "營收月增率在 60 日內的排名",
    },
    {
        "name": "revenue_stability",
        "display_name": "營收穩定性",
        "category": "revenue",
        "expression": "Mean($revenue, 63) / (Std($revenue, 63) + 1e-8)",
        "description": "營收均值 / 營收標準差",
    },
]


def seed_factors(session: Session, force: bool = False) -> int:
    """
    插入預設因子

    Args:
        session: 資料庫 Session
        force: 是否強制重新插入（會先清空現有因子）

    Returns:
        插入的因子數量
    """
    repo = FactorRepository(session)

    # 若不強制且已有資料，跳過
    existing = repo.get_all()
    if not force and existing:
        return 0

    # 強制模式：先刪除所有現有因子
    if force:
        for factor in existing:
            repo.delete(factor.id)

    # 插入預設因子
    count = 0
    for factor_data in DEFAULT_FACTORS:
        # 檢查是否已存在
        if repo.get_by_name(factor_data["name"]):
            continue
        repo.create(
            name=factor_data["name"],
            display_name=factor_data.get("display_name"),
            category=factor_data.get("category", "technical"),
            expression=factor_data["expression"],
            description=factor_data.get("description"),
        )
        count += 1

    return count


class FactorRepository:
    """因子定義存取"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        name: str,
        expression: str,
        display_name: str | None = None,
        category: str = "technical",
        description: str | None = None,
    ) -> Factor:
        """建立因子"""
        factor = Factor(
            name=name,
            display_name=display_name,
            category=category,
            expression=expression,
            description=description,
        )
        self._session.add(factor)
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def get_by_id(self, factor_id: int) -> Factor | None:
        """依 ID 取得因子"""
        stmt = select(Factor).where(Factor.id == factor_id)
        return self._session.execute(stmt).scalar()

    def get_by_name(self, name: str) -> Factor | None:
        """依名稱取得因子"""
        stmt = select(Factor).where(Factor.name == name)
        return self._session.execute(stmt).scalar()

    def get_enabled(self) -> list[Factor]:
        """取得所有啟用的因子"""
        stmt = select(Factor).where(Factor.enabled == True)
        return list(self._session.execute(stmt).scalars().all())

    def get_all(self, category: str | None = None, enabled: bool | None = None) -> list[Factor]:
        """取得所有因子（可篩選）"""
        stmt = select(Factor)
        if category is not None:
            stmt = stmt.where(Factor.category == category)
        if enabled is not None:
            stmt = stmt.where(Factor.enabled == enabled)
        return list(self._session.execute(stmt).scalars().all())

    def update(
        self,
        factor_id: int,
        name: str | None = None,
        display_name: str | None = None,
        category: str | None = None,
        expression: str | None = None,
        description: str | None = None,
    ) -> Factor | None:
        """更新因子"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return None
        if name is not None:
            factor.name = name
        if display_name is not None:
            factor.display_name = display_name
        if category is not None:
            factor.category = category
        if expression is not None:
            factor.expression = expression
        if description is not None:
            factor.description = description
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def delete(self, factor_id: int) -> bool:
        """刪除因子"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return False
        self._session.delete(factor)
        self._session.commit()
        return True

    def toggle(self, factor_id: int) -> Factor | None:
        """切換因子啟用狀態"""
        factor = self.get_by_id(factor_id)
        if not factor:
            return None
        factor.enabled = not factor.enabled
        self._session.commit()
        self._session.refresh(factor)
        return factor

    def get_selection_stats(self, factor_id: int) -> dict:
        """取得因子入選統計"""
        # 計算 times_evaluated（參與過的訓練次數）
        evaluated_stmt = (
            select(func.count())
            .select_from(TrainingFactorResult)
            .where(TrainingFactorResult.factor_id == factor_id)
        )
        times_evaluated = self._session.execute(evaluated_stmt).scalar() or 0

        # 計算 times_selected（被選中的次數）
        selected_stmt = (
            select(func.count())
            .select_from(TrainingFactorResult)
            .where(
                TrainingFactorResult.factor_id == factor_id,
                TrainingFactorResult.selected == True,
            )
        )
        times_selected = self._session.execute(selected_stmt).scalar() or 0

        selection_rate = times_selected / times_evaluated if times_evaluated > 0 else 0.0

        return {
            "times_evaluated": times_evaluated,
            "times_selected": times_selected,
            "selection_rate": selection_rate,
        }

    def get_selection_history(self, factor_id: int) -> list[dict]:
        """取得因子入選歷史"""
        from src.repositories.models import TrainingRun

        stmt = (
            select(TrainingFactorResult, TrainingRun)
            .join(TrainingRun)
            .where(TrainingFactorResult.factor_id == factor_id)
            .order_by(TrainingRun.started_at.desc())
        )
        results = self._session.execute(stmt).all()
        return [
            {
                "model_id": f"m{r.TrainingRun.id:03d}",
                "trained_at": r.TrainingRun.started_at.date().isoformat(),
                "selected": r.TrainingFactorResult.selected,
            }
            for r in results
        ]
