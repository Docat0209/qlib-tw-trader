"""週期計算工具"""

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import TradingCalendar
from src.shared.constants import EMBARGO_DAYS, TRAIN_DAYS, VALID_DAYS


@dataclass
class WeekSlot:
    """週訓練時段"""

    week_id: str  # "2026W05"
    valid_end: date  # 該週最後一個交易日
    valid_start: date
    train_end: date
    train_start: date


def compute_week_id(d: date) -> str:
    """
    計算 ISO 週 ID

    格式：{YYYY}W{WW}
    例：2026W05（2026 年第 5 週）
    """
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def parse_week_id(week_id: str) -> tuple[int, int]:
    """
    解析週 ID

    "2026W05" -> (2026, 5)
    """
    # 格式: YYYYWww
    year = int(week_id[:4])
    week = int(week_id[5:])
    return year, week


def get_week_friday(week_id: str) -> date:
    """
    從週 ID 取得該週週五的日期

    ISO 週：週一是第 1 天，週五是第 5 天
    """
    year, week = parse_week_id(week_id)
    # ISO 週的週一
    # 使用 date.fromisocalendar (Python 3.8+)
    monday = date.fromisocalendar(year, week, 1)
    friday = monday + timedelta(days=4)
    return friday


def get_week_valid_end(week_id: str, session: Session) -> date:
    """
    從週 ID 取得 valid_end（該週最後一個交易日）

    通常是週五，但如果週五不是交易日，則取該週最後一個交易日
    """
    year, week = parse_week_id(week_id)
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)

    # 查詢該週的交易日（由新到舊）
    stmt = (
        select(TradingCalendar.date)
        .where(TradingCalendar.date >= monday)
        .where(TradingCalendar.date <= sunday)
        .where(TradingCalendar.is_trading_day == True)
        .order_by(TradingCalendar.date.desc())
        .limit(1)
    )
    result = session.execute(stmt).scalar()

    if result:
        return result

    # 如果該週沒有交易日（例如春節），返回週五作為佔位
    return monday + timedelta(days=4)


def get_trading_day_offset(
    base_date: date,
    offset: int,
    session: Session,
) -> date:
    """
    取得相對於 base_date 偏移 N 個交易日的日期

    offset > 0: 往後（未來）
    offset < 0: 往前（過去）
    offset = 0: 返回 base_date（如果是交易日）或最近的交易日
    """
    if offset == 0:
        # 檢查 base_date 是否為交易日
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.date == base_date)
            .where(TradingCalendar.is_trading_day == True)
        )
        if session.execute(stmt).scalar():
            return base_date
        # 不是交易日，找最近的過去交易日
        offset = -1

    if offset > 0:
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.date > base_date)
            .where(TradingCalendar.is_trading_day == True)
            .order_by(TradingCalendar.date)
            .limit(offset)
        )
        dates = list(session.execute(stmt).scalars().all())
        return dates[-1] if dates else base_date + timedelta(days=offset)
    else:
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.date < base_date)
            .where(TradingCalendar.is_trading_day == True)
            .order_by(TradingCalendar.date.desc())
            .limit(abs(offset))
        )
        dates = list(session.execute(stmt).scalars().all())
        return dates[-1] if dates else base_date + timedelta(days=offset)


def compute_factor_pool_hash(factor_ids: list[int]) -> str:
    """
    計算因子池 hash（MD5 前 6 位）

    用於識別不同的因子池版本
    """
    sorted_ids = sorted(factor_ids)
    content = ",".join(str(fid) for fid in sorted_ids)
    return hashlib.md5(content.encode()).hexdigest()[:6]


@dataclass
class WeekSlotWithStatus(WeekSlot):
    """週訓練時段（含可訓練狀態）"""

    is_trainable: bool = True  # 是否有足夠資料可訓練


def get_trainable_weeks(
    data_start: date,
    data_end: date,
    session: Session,
    train_days: int = TRAIN_DAYS,
    valid_days: int = VALID_DAYS,
    embargo_days: int = EMBARGO_DAYS,
    lookback_buffer: int = 60,
    include_insufficient: bool = False,
) -> list[WeekSlot] | list[WeekSlotWithStatus]:
    """
    計算所有可訓練的週

    規則：valid_end 必須滿足
    - valid_end <= data_end
    - train_start - lookback_buffer >= data_start（因子計算需要回看期）

    Args:
        data_start: 資料庫資料開始日期
        data_end: 資料庫資料結束日期
        session: 資料庫連線
        train_days: 訓練期交易日數
        valid_days: 驗證期交易日數
        embargo_days: Embargo 期交易日數
        lookback_buffer: 因子計算需要的回看期（日曆日）
        include_insufficient: 是否包含資料不足的週（標記為 is_trainable=False）

    Returns:
        可訓練週列表（最新在前）
    """
    # 取得所有交易日
    stmt = (
        select(TradingCalendar.date)
        .where(TradingCalendar.date >= data_start)
        .where(TradingCalendar.date <= data_end)
        .where(TradingCalendar.is_trading_day == True)
        .order_by(TradingCalendar.date)
    )
    all_trading_days = list(session.execute(stmt).scalars().all())

    if len(all_trading_days) < train_days + embargo_days + valid_days:
        return []

    # 建立交易日索引
    trading_day_index = {d: i for i, d in enumerate(all_trading_days)}

    # 枚舉日期範圍內所有的週（包含無交易日的週）
    all_week_ids = set()
    current = data_start
    while current <= data_end:
        all_week_ids.add(compute_week_id(current))
        current += timedelta(days=1)

    # 收集所有週
    slots: list[WeekSlotWithStatus] = []

    for week_id in all_week_ids:
        # 計算該週的 valid_end（該週最後一個交易日）
        actual_valid_end = get_week_valid_end(week_id, session)

        # 如果 actual_valid_end 超出資料範圍，跳過（這週還沒到）
        if actual_valid_end > data_end:
            continue

        # 檢查 valid_end 是否在交易日索引中
        if actual_valid_end not in trading_day_index:
            # 該週沒有交易日（如農曆新年），標記為資料不足
            if include_insufficient:
                # 取得該週週五作為顯示用日期
                friday = get_week_friday(week_id)
                slots.append(
                    WeekSlotWithStatus(
                        week_id=week_id,
                        valid_end=friday,
                        valid_start=friday,
                        train_end=data_start,
                        train_start=data_start,
                        is_trainable=False,
                    )
                )
            continue

        valid_end_idx = trading_day_index[actual_valid_end]

        # 計算各期間的索引
        valid_start_idx = valid_end_idx - valid_days + 1
        train_end_idx = valid_start_idx - embargo_days - 1
        train_start_idx = train_end_idx - train_days + 1

        # 檢查是否有足夠資料
        is_trainable = True
        if train_start_idx < 0 or valid_start_idx < 0:
            is_trainable = False
            if not include_insufficient:
                continue
            # 使用預設日期填充
            train_start = data_start
            train_end = data_start
            valid_start = all_trading_days[max(0, valid_start_idx)] if valid_start_idx >= 0 else actual_valid_end
        else:
            train_start = all_trading_days[train_start_idx]
            train_end = all_trading_days[train_end_idx]
            valid_start = all_trading_days[valid_start_idx]

            # 檢查 lookback buffer
            if train_start - timedelta(days=lookback_buffer) < data_start:
                is_trainable = False
                if not include_insufficient:
                    continue

        slots.append(
            WeekSlotWithStatus(
                week_id=week_id,
                valid_end=actual_valid_end,
                valid_start=valid_start,
                train_end=train_end,
                train_start=train_start,
                is_trainable=is_trainable,
            )
        )

    # 按 week_id 降序排列（最新在前）
    slots.sort(key=lambda s: s.week_id, reverse=True)

    if include_insufficient:
        return slots
    else:
        # 返回原始 WeekSlot 類型以保持向後兼容
        return [
            WeekSlot(
                week_id=s.week_id,
                valid_end=s.valid_end,
                valid_start=s.valid_start,
                train_end=s.train_end,
                train_start=s.train_start,
            )
            for s in slots
        ]


def get_current_week_id() -> str:
    """取得當前週 ID"""
    return compute_week_id(date.today())
