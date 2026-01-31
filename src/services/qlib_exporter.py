"""
Qlib .bin 導出服務

將 SQLite 資料庫中的資料轉換為 Qlib 可讀取的二進位格式。
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.daily import (
    AdjCloseRepository,
    InstitutionalRepository,
    MarginRepository,
    OHLCVRepository,
    PERRepository,
    SecuritiesLendingRepository,
    ShareholdingRepository,
)
from src.repositories.models import StockMonthlyRevenue, StockUniverse, TradingCalendar
from src.repositories.periodic import MonthlyRevenueRepository


@dataclass
class ExportConfig:
    """導出配置"""

    start_date: date
    end_date: date
    output_dir: Path
    include_fields: list[str] | None = None  # None = 全部


@dataclass
class ExportResult:
    """導出結果"""

    stocks_exported: int
    fields_per_stock: int
    total_files: int
    calendar_days: int
    output_path: str
    errors: list[dict]


class QlibExporter:
    """Qlib .bin 導出器"""

    # 可導出的欄位定義：(資料來源, 屬性名稱)
    DAILY_FIELDS = {
        # OHLCV（5 個）
        "open": ("ohlcv", "open"),
        "high": ("ohlcv", "high"),
        "low": ("ohlcv", "low"),
        "close": ("ohlcv", "close"),
        "volume": ("ohlcv", "volume"),
        # 還原股價（1 個）
        "adj_close": ("adj", "adj_close"),
        # 估值（3 個）
        "pe_ratio": ("per", "pe_ratio"),
        "pb_ratio": ("per", "pb_ratio"),
        "dividend_yield": ("per", "dividend_yield"),
        # 三大法人（6 個）
        "foreign_buy": ("institutional", "foreign_buy"),
        "foreign_sell": ("institutional", "foreign_sell"),
        "trust_buy": ("institutional", "trust_buy"),
        "trust_sell": ("institutional", "trust_sell"),
        "dealer_buy": ("institutional", "dealer_buy"),
        "dealer_sell": ("institutional", "dealer_sell"),
        # 融資融券（6 個）
        "margin_buy": ("margin", "margin_buy"),
        "margin_sell": ("margin", "margin_sell"),
        "margin_balance": ("margin", "margin_balance"),
        "short_buy": ("margin", "short_buy"),
        "short_sell": ("margin", "short_sell"),
        "short_balance": ("margin", "short_balance"),
        # 外資持股（7 個）
        "total_shares": ("shareholding", "total_shares"),
        "foreign_shares": ("shareholding", "foreign_shares"),
        "foreign_ratio": ("shareholding", "foreign_ratio"),
        "foreign_remaining_shares": ("shareholding", "foreign_remaining_shares"),
        "foreign_remaining_ratio": ("shareholding", "foreign_remaining_ratio"),
        "foreign_upper_limit_ratio": ("shareholding", "foreign_upper_limit_ratio"),
        "chinese_upper_limit_ratio": ("shareholding", "chinese_upper_limit_ratio"),
        # 借券（1 個）
        "lending_volume": ("lending", "lending_volume"),
        # 月營收 PIT（1 個）
        "revenue": ("revenue_pit", "revenue"),
    }

    # 月營收公布日：次月 10 日
    REVENUE_ANNOUNCE_DAY = 10

    def __init__(self, session: Session):
        self._session = session
        self._init_repositories()

    def _init_repositories(self):
        """初始化 Repositories"""
        self._repos = {
            "ohlcv": OHLCVRepository(self._session),
            "adj": AdjCloseRepository(self._session),
            "per": PERRepository(self._session),
            "institutional": InstitutionalRepository(self._session),
            "margin": MarginRepository(self._session),
            "shareholding": ShareholdingRepository(self._session),
            "lending": SecuritiesLendingRepository(self._session),
            "revenue": MonthlyRevenueRepository(self._session),
        }

    def export(self, config: ExportConfig) -> ExportResult:
        """
        執行導出

        Args:
            config: 導出配置

        Returns:
            導出結果
        """
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 取得交易日曆
        calendar = self._get_trading_calendar(config.start_date, config.end_date)
        if not calendar:
            raise ValueError("No trading days found in the specified range")

        # 2. 產生 calendars/day.txt
        self._write_calendar(output_dir / "calendars", calendar)

        # 3. 取得股票池
        stocks = self._get_stock_universe()
        if not stocks:
            raise ValueError("Stock universe is empty")

        # 4. 決定要導出的欄位
        fields = config.include_fields or list(self.DAILY_FIELDS.keys())

        # 5. 逐股導出
        errors = []
        stocks_exported = 0
        total_files = 0

        for stock in stocks:
            stock_id = stock.stock_id
            try:
                files_written = self._export_stock(
                    stock_id=stock_id,
                    calendar=calendar,
                    fields=fields,
                    output_dir=output_dir / "features" / stock_id,
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
                stocks_exported += 1
                total_files += files_written
            except Exception as e:
                errors.append({"stock_id": stock_id, "error": str(e)})

        # 6. 產生 instruments/all.txt
        self._write_instruments(
            output_dir / "instruments",
            [s.stock_id for s in stocks],
            config.start_date,
            config.end_date,
        )

        return ExportResult(
            stocks_exported=stocks_exported,
            fields_per_stock=len(fields),
            total_files=total_files,
            calendar_days=len(calendar),
            output_path=str(output_dir),
            errors=errors,
        )

    def _get_trading_calendar(self, start_date: date, end_date: date) -> list[date]:
        """取得交易日曆"""
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.date >= start_date)
            .where(TradingCalendar.date <= end_date)
            .where(TradingCalendar.is_trading_day == True)
            .order_by(TradingCalendar.date)
        )
        return list(self._session.execute(stmt).scalars().all())

    def _get_stock_universe(self) -> list[StockUniverse]:
        """取得股票池"""
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        return list(self._session.execute(stmt).scalars().all())

    def _write_calendar(self, calendar_dir: Path, dates: list[date]):
        """寫入交易日曆"""
        calendar_dir.mkdir(parents=True, exist_ok=True)
        with open(calendar_dir / "day.txt", "w") as f:
            for d in dates:
                f.write(f"{d.strftime('%Y-%m-%d')}\n")

    def _write_instruments(
        self,
        instruments_dir: Path,
        stock_ids: list[str],
        start_date: date,
        end_date: date,
    ):
        """寫入股票清單"""
        instruments_dir.mkdir(parents=True, exist_ok=True)
        with open(instruments_dir / "all.txt", "w") as f:
            for stock_id in stock_ids:
                f.write(f"{stock_id}\t{start_date}\t{end_date}\n")

    def _export_stock(
        self,
        stock_id: str,
        calendar: list[date],
        fields: list[str],
        output_dir: Path,
        start_date: date,
        end_date: date,
    ) -> int:
        """
        導出單一股票的所有欄位

        Qlib .bin 格式：
        - 開頭 4 bytes: start_index (float32) - 資料在日曆中的起始位置
        - 之後: 資料陣列 (float32)

        Returns:
            寫入的檔案數
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立日期索引
        date_to_idx = {d: i for i, d in enumerate(calendar)}
        n_days = len(calendar)

        # 預先載入所有資料
        data_cache = self._load_stock_data(stock_id, start_date, end_date)

        # 找出該股票有資料的第一個日期索引
        start_index = self._find_start_index(data_cache, date_to_idx)

        files_written = 0
        for field in fields:
            if field not in self.DAILY_FIELDS:
                continue

            source, attr = self.DAILY_FIELDS[field]

            # 建立空陣列（填充 NaN）
            arr = np.full(n_days, np.nan, dtype=np.float32)

            if source == "revenue_pit":
                # 月營收 PIT 處理
                arr = self._expand_revenue_pit(
                    data_cache.get("revenue", []),
                    calendar,
                )
            else:
                # 一般日頻資料
                records = data_cache.get(source, [])
                for rec in records:
                    if rec.date in date_to_idx:
                        idx = date_to_idx[rec.date]
                        value = getattr(rec, attr, None)
                        if value is not None:
                            arr[idx] = float(value)

            # 寫入 .bin 檔（qlib 格式：start_index + data）
            bin_path = output_dir / f"{field}.day.bin"
            with open(bin_path, "wb") as f:
                # 先寫入 start_index
                np.array([start_index], dtype="<f").tofile(f)
                # 再寫入資料（從 start_index 開始）
                arr[start_index:].astype("<f").tofile(f)
            files_written += 1

        return files_written

    def _find_start_index(
        self,
        data_cache: dict,
        date_to_idx: dict[date, int],
    ) -> int:
        """找出該股票有資料的第一個日期索引"""
        min_idx = float("inf")

        # 從 OHLCV 找最早日期
        ohlcv_records = data_cache.get("ohlcv", [])
        if ohlcv_records:
            for rec in ohlcv_records:
                if rec.date in date_to_idx:
                    min_idx = min(min_idx, date_to_idx[rec.date])
                    break  # 已排序，第一個就是最早的

        return int(min_idx) if min_idx != float("inf") else 0

    def _load_stock_data(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> dict:
        """載入股票的所有資料"""
        return {
            "ohlcv": self._repos["ohlcv"].get(stock_id, start_date, end_date),
            "adj": self._repos["adj"].get(stock_id, start_date, end_date),
            "per": self._repos["per"].get(stock_id, start_date, end_date),
            "institutional": self._repos["institutional"].get(
                stock_id, start_date, end_date
            ),
            "margin": self._repos["margin"].get(stock_id, start_date, end_date),
            "shareholding": self._repos["shareholding"].get(
                stock_id, start_date, end_date
            ),
            "lending": self._repos["lending"].get(stock_id, start_date, end_date),
            "revenue": self._load_revenue(stock_id, start_date, end_date),
        }

    def _load_revenue(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> list:
        """載入月營收（含 PIT 所需的額外月份）"""
        # 需要往前多取一個月（因為公布延遲）
        start_year = start_date.year
        start_month = start_date.month - 1
        if start_month < 1:
            start_month = 12
            start_year -= 1

        return self._repos["revenue"].get(
            stock_id,
            start_year,
            start_month,
            end_date.year,
            end_date.month,
        )

    def _expand_revenue_pit(
        self,
        revenue_records: list,
        calendar: list[date],
    ) -> np.ndarray:
        """
        將月營收展開為日頻（PIT 格式）

        規則：
        - N 月營收於 N+1 月 10 日公布
        - 從公布日起 forward fill 至下一個公布日
        """
        n_days = len(calendar)
        arr = np.full(n_days, np.nan, dtype=np.float32)

        if not revenue_records:
            return arr

        # 建立 (公布日, 營收) 對照表
        announce_map = {}
        for rec in revenue_records:
            # 計算公布日（次月 10 日）
            announce_year = rec.year
            announce_month = rec.month + 1
            if announce_month > 12:
                announce_month = 1
                announce_year += 1
            announce_date = date(announce_year, announce_month, self.REVENUE_ANNOUNCE_DAY)
            announce_map[announce_date] = float(rec.revenue)

        # 按公布日排序
        sorted_dates = sorted(announce_map.keys())

        # Forward fill
        current_value = np.nan
        sorted_idx = 0

        for i, cal_date in enumerate(calendar):
            # 檢查是否有新公布
            while sorted_idx < len(sorted_dates) and sorted_dates[sorted_idx] <= cal_date:
                current_value = announce_map[sorted_dates[sorted_idx]]
                sorted_idx += 1

            arr[i] = current_value

        return arr

    def get_available_fields(self) -> list[dict]:
        """取得所有可導出的欄位"""
        return [
            {"name": name, "source": source, "attribute": attr}
            for name, (source, attr) in self.DAILY_FIELDS.items()
        ]
