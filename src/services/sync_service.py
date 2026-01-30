"""
資料同步服務
"""

import os
from datetime import date, timedelta
from decimal import Decimal

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

import yfinance as yf

from src.repositories.models import StockDaily, StockDailyAdj, StockDailyInstitutional, StockDailyMargin, StockDailyPER, StockDailySecuritiesLending, StockDailyShareholding, StockDividend, StockMonthlyRevenue, StockQuarterlyBalance, StockQuarterlyCashFlow, StockQuarterlyFinancial, StockUniverse, TradingCalendar


class SyncService:
    """資料同步服務"""

    FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
    TWSE_RWD_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
    TWSE_PER_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_ALL"
    TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
    TWSE_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

    # 最低資料筆數門檻（低於此值視為不完整）
    MIN_DAILY_RECORDS = 100      # 日頻資料
    MIN_MONTHLY_RECORDS = 12     # 月營收（至少 1 年）
    MIN_QUARTERLY_RECORDS = 8    # 季度財報（至少 2 年）

    @staticmethod
    def _calc_coverage(total: int, expected: int, min_records: int) -> float:
        """
        計算覆蓋率，考慮最低筆數門檻
        - 如果 total < min_records，覆蓋率會被壓低
        - 公式：min(基於期望的覆蓋率, 基於最低門檻的覆蓋率)
        """
        if total == 0 or expected == 0:
            return 0.0
        coverage_by_expected = (total / expected) * 100
        coverage_by_min = (total / min_records) * 100
        return min(coverage_by_expected, coverage_by_min)

    def __init__(self, session: Session):
        self._session = session
        self._finmind_token = os.getenv("FINMIND_KEY", "")
        # 調試：記錄 token 狀態
        if self._finmind_token:
            print(f"[SyncService] FinMind token loaded: {self._finmind_token[:20]}...")
        else:
            print("[SyncService] WARNING: FINMIND_KEY not found in environment!")

    async def _fetch_finmind(self, dataset: str, params: dict, timeout: int = 60) -> dict:
        """統一的 FinMind API 呼叫"""
        request_params = {
            "dataset": dataset,
            **params,
        }
        if self._finmind_token:
            request_params["token"] = self._finmind_token
            print(f"[FinMind] Calling {dataset} with token")
        else:
            print(f"[FinMind] WARNING: Calling {dataset} WITHOUT token!")

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.FINMIND_URL, params=request_params, timeout=timeout)
            print(f"[FinMind] Response status: {resp.status_code}")

            # 嘗試解析 JSON（即使是錯誤狀態碼）
            try:
                data = resp.json()
            except Exception:
                data = {}

            # 檢查 HTTP 狀態碼
            if resp.status_code == 402:
                msg = data.get("msg", "Rate limit exceeded")
                raise RuntimeError(f"FinMind API 速率限制: {msg}（免費版每小時 600 次，請稍後再試或升級方案）")

            if resp.status_code != 200:
                msg = data.get("msg", f"HTTP {resp.status_code}")
                raise RuntimeError(f"FinMind API error: {msg}")

        # 檢查 API 回應狀態
        if data.get("status") == 402:
            msg = data.get("msg", "Rate limit exceeded")
            raise RuntimeError(f"FinMind API 速率限制: {msg}（免費版每小時 600 次，請稍後再試或升級方案）")

        if data.get("status") != 200:
            raise RuntimeError(f"FinMind API error: {data.get('msg', 'Unknown error')}")

        return data

    # =========================================================================
    # 交易日曆
    # =========================================================================

    async def sync_trading_calendar(self, start_date: date, end_date: date) -> int:
        """
        同步交易日曆（用 0050 ETF 推算交易日，使用 yfinance）
        Returns: 新增的交易日數量
        """
        # 從 yfinance 取得 0050 的交易日
        ticker = yf.Ticker("0050.TW")
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
        )

        if df.empty:
            return 0

        trading_dates = {idx.date() for idx in df.index}

        # 取得已存在的交易日
        stmt = select(TradingCalendar.date).where(
            TradingCalendar.date >= start_date,
            TradingCalendar.date <= end_date,
        )
        existing = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 新增缺少的交易日
        new_dates = trading_dates - existing
        for d in new_dates:
            self._session.add(TradingCalendar(date=d, is_trading_day=True))

        self._session.commit()
        return len(new_dates)

    def get_trading_dates(self, start_date: date, end_date: date) -> list[date]:
        """取得指定區間的交易日"""
        stmt = (
            select(TradingCalendar.date)
            .where(
                TradingCalendar.date >= start_date,
                TradingCalendar.date <= end_date,
                TradingCalendar.is_trading_day == True,
            )
            .order_by(TradingCalendar.date)
        )
        return [row[0] for row in self._session.execute(stmt).fetchall()]

    def get_latest_trading_date(self) -> date | None:
        """取得最新交易日"""
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.is_trading_day == True)
            .order_by(TradingCalendar.date.desc())
            .limit(1)
        )
        result = self._session.execute(stmt).fetchone()
        return result[0] if result else None

    def get_previous_trading_date(self) -> date | None:
        """取得今天之前的最新交易日（用於 bulk sync）"""
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.is_trading_day == True)
            .where(TradingCalendar.date < date.today())
            .order_by(TradingCalendar.date.desc())
            .limit(1)
        )
        result = self._session.execute(stmt).fetchone()
        return result[0] if result else None

    def get_recent_trading_dates(self, days: int = 7) -> list[date]:
        """取得最近 N 個交易日（不含今天），由舊到新排序"""
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.is_trading_day == True)
            .where(TradingCalendar.date < date.today())
            .order_by(TradingCalendar.date.desc())
            .limit(days)
        )
        dates = [row[0] for row in self._session.execute(stmt).fetchall()]
        return sorted(dates)  # 由舊到新

    def count_trading_days(self, start_date: date, end_date: date) -> int:
        """計算指定區間的交易日數"""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(TradingCalendar).where(
            TradingCalendar.date >= start_date,
            TradingCalendar.date <= end_date,
            TradingCalendar.is_trading_day == True,
        )
        return self._session.execute(stmt).scalar() or 0

    # =========================================================================
    # 股票日K線
    # =========================================================================

    async def sync_stock_daily(
        self,
        stock_id: str,
        start_date: date,
        end_date: date,
    ) -> dict:
        """
        同步單一股票的日K線
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDaily.date).where(
            StockDaily.stock_id == stock_id,
            StockDaily.date >= start_date,
            StockDaily.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockPrice", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            # 解析資料
            open_val = self._safe_decimal(r.get("open"))
            close = self._safe_decimal(r.get("close"))
            if open_val is None or close is None:
                continue

            self._session.add(
                StockDaily(
                    stock_id=stock_id,
                    date=r_date,
                    open=open_val,
                    high=self._safe_decimal(r.get("max")) or open_val,
                    low=self._safe_decimal(r.get("min")) or open_val,
                    close=close,
                    volume=self._safe_int(r.get("Trading_Volume")),
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期（可能該股票當天停牌）
        stmt = select(StockDaily.date).where(
            StockDaily.stock_id == stock_id,
            StockDaily.date >= start_date,
            StockDaily.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    async def sync_stock_daily_bulk(self, target_date: date) -> dict:
        """
        同步全市場當日資料（用 TWSE RWD bulk API）
        只儲存股票池內的股票
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id)
        universe = {row[0] for row in self._session.execute(stmt).fetchall()}

        if not universe:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 呼叫 TWSE RWD API
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.TWSE_RWD_URL, timeout=30)
            data = resp.json()

        if data.get("stat") != "OK":
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "TWSE API failed"}

        # 檢查日期
        data_date_str = data.get("date", "")
        if data_date_str:
            data_date = date(
                int(data_date_str[:4]),
                int(data_date_str[4:6]),
                int(data_date_str[6:8]),
            )
            if data_date != target_date:
                return {
                    "date": target_date.isoformat(),
                    "total": 0,
                    "inserted": 0,
                    "error": f"Data date mismatch: {data_date} != {target_date}",
                }

        # 確保交易日曆有這天
        existing_cal = self._session.get(TradingCalendar, target_date)
        if not existing_cal:
            self._session.add(TradingCalendar(date=target_date, is_trading_day=True))

        # 取得已有資料
        stmt = select(StockDaily.stock_id).where(StockDaily.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        rows = data.get("data", [])
        inserted = 0

        for row in rows:
            if len(row) < 8:
                continue

            stock_id = row[0].strip()

            # 只儲存股票池內的股票
            if stock_id not in universe:
                continue
            # 跳過已存在的
            if stock_id in existing_stocks:
                continue

            open_val = self._safe_decimal(row[4])
            close = self._safe_decimal(row[7])
            if open_val is None or close is None:
                continue

            self._session.add(
                StockDaily(
                    stock_id=stock_id,
                    date=target_date,
                    open=open_val,
                    high=self._safe_decimal(row[5]) or open_val,
                    low=self._safe_decimal(row[6]) or open_val,
                    close=close,
                    volume=self._safe_int(row[2]),
                )
            )
            inserted += 1

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(rows),
            "inserted": inserted,
        }

    async def sync_all_stocks(self, start_date: date, end_date: date) -> dict:
        """
        同步股票池內所有股票的日K線
        Returns: {"stocks": int, "total_inserted": int, "errors": list}
        """
        # 先同步交易日曆
        await self.sync_trading_calendar(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]

        total_inserted = 0
        errors = []

        for stock_id in stock_ids:
            try:
                result = await self.sync_stock_daily(stock_id, start_date, end_date)
                total_inserted += result["inserted"]
            except Exception as e:
                errors.append({"stock_id": stock_id, "error": str(e)})

        return {
            "stocks": len(stock_ids),
            "total_inserted": total_inserted,
            "errors": errors,
        }

    # =========================================================================
    # PER/PBR/殖利率
    # =========================================================================

    async def sync_per_bulk(self, target_date: date) -> dict:
        """
        同步全市場 PER/PBR/殖利率（用 TWSE RWD bulk API）
        只儲存股票池內的股票
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id)
        universe = {row[0] for row in self._session.execute(stmt).fetchall()}

        if not universe:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 呼叫 TWSE RWD API
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.TWSE_PER_URL, timeout=30)
            data = resp.json()

        if data.get("stat") != "OK":
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "TWSE API failed"}

        # 檢查日期
        data_date_str = data.get("date", "")
        if data_date_str:
            data_date = date(
                int(data_date_str[:4]),
                int(data_date_str[4:6]),
                int(data_date_str[6:8]),
            )
            if data_date != target_date:
                return {
                    "date": target_date.isoformat(),
                    "total": 0,
                    "inserted": 0,
                    "error": f"Data date mismatch: {data_date} != {target_date}",
                }

        # 取得已有資料
        stmt = select(StockDailyPER.stock_id).where(StockDailyPER.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        rows = data.get("data", [])
        inserted = 0

        for row in rows:
            if len(row) < 5:
                continue

            stock_id = row[0].strip()

            # 只儲存股票池內的股票
            if stock_id not in universe:
                continue
            # 跳過已存在的
            if stock_id in existing_stocks:
                continue

            # BWIBBU_ALL 欄位: [股票代號, 股票名稱, 本益比, 殖利率(%), 股價淨值比]
            pe_ratio = self._safe_decimal(row[2])
            dividend_yield = self._safe_decimal(row[3])
            pb_ratio = self._safe_decimal(row[4])

            # 至少要有一個值
            if pe_ratio is None and pb_ratio is None and dividend_yield is None:
                continue

            self._session.add(
                StockDailyPER(
                    stock_id=stock_id,
                    date=target_date,
                    pe_ratio=pe_ratio,
                    pb_ratio=pb_ratio,
                    dividend_yield=dividend_yield,
                )
            )
            inserted += 1

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(rows),
            "inserted": inserted,
        }

    async def sync_per(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的 PER/PBR/殖利率（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailyPER.date).where(
            StockDailyPER.stock_id == stock_id,
            StockDailyPER.date >= start_date,
            StockDailyPER.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockPER", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            pe_ratio = self._safe_decimal(r.get("PER"))
            pb_ratio = self._safe_decimal(r.get("PBR"))
            dividend_yield = self._safe_decimal(r.get("dividend_yield"))

            if pe_ratio is None and pb_ratio is None and dividend_yield is None:
                continue

            self._session.add(
                StockDailyPER(
                    stock_id=stock_id,
                    date=r_date,
                    pe_ratio=pe_ratio,
                    pb_ratio=pb_ratio,
                    dividend_yield=dividend_yield,
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailyPER.date).where(
            StockDailyPER.stock_id == stock_id,
            StockDailyPER.date >= start_date,
            StockDailyPER.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    def get_per_status(self, start_date: date, end_date: date) -> dict:
        """取得 PER 資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailyPER.date),
                func.max(StockDailyPER.date),
                func.count(),
            ).where(
                StockDailyPER.stock_id == stock.stock_id,
                StockDailyPER.date >= start_date,
                StockDailyPER.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 三大法人買賣超
    # =========================================================================

    async def sync_institutional_bulk(self, target_date: date) -> dict:
        """
        同步全市場三大法人買賣超（用 TWSE RWD API）
        只儲存股票池內的股票
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id)
        universe = {row[0] for row in self._session.execute(stmt).fetchall()}

        if not universe:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 呼叫 TWSE RWD API（需帶日期參數）
        params = {
            "date": target_date.strftime("%Y%m%d"),
            "selectType": "ALL",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.TWSE_INSTITUTIONAL_URL, params=params, timeout=30)
            data = resp.json()

        if data.get("stat") != "OK":
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "TWSE API failed"}

        # 取得已有資料
        stmt = select(StockDailyInstitutional.stock_id).where(StockDailyInstitutional.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        rows = data.get("data", [])
        inserted = 0

        for row in rows:
            if len(row) < 14:
                continue

            stock_id = row[0].strip()

            # 只儲存股票池內的股票
            if stock_id not in universe:
                continue
            # 跳過已存在的
            if stock_id in existing_stocks:
                continue

            # T86 欄位:
            # 0: 證券代號
            # 2: 外陸資買進股數(不含外資自營商)
            # 3: 外陸資賣出股數(不含外資自營商)
            # 8: 投信買進股數
            # 9: 投信賣出股數
            # 12: 自營商買進股數(自行買賣)
            # 13: 自營商賣出股數(自行買賣)
            foreign_buy = self._safe_int(row[2])
            foreign_sell = self._safe_int(row[3])
            trust_buy = self._safe_int(row[8])
            trust_sell = self._safe_int(row[9])
            dealer_buy = self._safe_int(row[12])
            dealer_sell = self._safe_int(row[13])

            self._session.add(
                StockDailyInstitutional(
                    stock_id=stock_id,
                    date=target_date,
                    foreign_buy=foreign_buy,
                    foreign_sell=foreign_sell,
                    trust_buy=trust_buy,
                    trust_sell=trust_sell,
                    dealer_buy=dealer_buy,
                    dealer_sell=dealer_sell,
                )
            )
            inserted += 1

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(rows),
            "inserted": inserted,
        }

    async def sync_institutional(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的三大法人買賣超（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailyInstitutional.date).where(
            StockDailyInstitutional.stock_id == stock_id,
            StockDailyInstitutional.date >= start_date,
            StockDailyInstitutional.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])

        # FinMind 回傳每個法人一筆，需要彙總成一筆
        grouped = {}
        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            if r_date not in grouped:
                grouped[r_date] = {
                    "foreign_buy": 0, "foreign_sell": 0,
                    "trust_buy": 0, "trust_sell": 0,
                    "dealer_buy": 0, "dealer_sell": 0,
                }

            inv_type = r.get("name", "")
            buy = self._safe_int(r.get("buy", 0))
            sell = self._safe_int(r.get("sell", 0))

            if "外資" in inv_type or "外陸資" in inv_type:
                grouped[r_date]["foreign_buy"] += buy
                grouped[r_date]["foreign_sell"] += sell
            elif "投信" in inv_type:
                grouped[r_date]["trust_buy"] += buy
                grouped[r_date]["trust_sell"] += sell
            elif "自營商" in inv_type:
                grouped[r_date]["dealer_buy"] += buy
                grouped[r_date]["dealer_sell"] += sell

        inserted = 0
        for r_date, values in grouped.items():
            self._session.add(
                StockDailyInstitutional(
                    stock_id=stock_id,
                    date=r_date,
                    **values,
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailyInstitutional.date).where(
            StockDailyInstitutional.stock_id == stock_id,
            StockDailyInstitutional.date >= start_date,
            StockDailyInstitutional.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    def get_institutional_status(self, start_date: date, end_date: date) -> dict:
        """取得三大法人資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailyInstitutional.date),
                func.max(StockDailyInstitutional.date),
                func.count(),
            ).where(
                StockDailyInstitutional.stock_id == stock.stock_id,
                StockDailyInstitutional.date >= start_date,
                StockDailyInstitutional.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 融資融券
    # =========================================================================

    async def sync_margin_bulk(self, target_date: date) -> dict:
        """
        同步全市場融資融券（用 TWSE RWD API）
        只儲存股票池內的股票
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id)
        universe = {row[0] for row in self._session.execute(stmt).fetchall()}

        if not universe:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 呼叫 TWSE RWD API（需帶日期參數）
        params = {
            "date": target_date.strftime("%Y%m%d"),
            "selectType": "ALL",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.TWSE_MARGIN_URL, params=params, timeout=30)
            data = resp.json()

        if data.get("stat") != "OK":
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "TWSE API failed"}

        # 取得已有資料
        stmt = select(StockDailyMargin.stock_id).where(StockDailyMargin.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        # tables[1] 是融資融券彙總
        tables = data.get("tables", [])
        if len(tables) < 2:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "No margin data"}

        rows = tables[1].get("data", [])
        inserted = 0

        for row in rows:
            if len(row) < 13:
                continue

            stock_id = row[0].strip()

            # 只儲存股票池內的股票
            if stock_id not in universe:
                continue
            # 跳過已存在的
            if stock_id in existing_stocks:
                continue

            # MI_MARGN 欄位:
            # 融資: 買進[2], 賣出[3], 今日餘額[6]
            # 融券: 買進[8], 賣出[9], 今日餘額[12]
            margin_buy = self._safe_int(row[2])
            margin_sell = self._safe_int(row[3])
            margin_balance = self._safe_int(row[6])
            short_buy = self._safe_int(row[8])
            short_sell = self._safe_int(row[9])
            short_balance = self._safe_int(row[12])

            self._session.add(
                StockDailyMargin(
                    stock_id=stock_id,
                    date=target_date,
                    margin_buy=margin_buy,
                    margin_sell=margin_sell,
                    margin_balance=margin_balance,
                    short_buy=short_buy,
                    short_sell=short_sell,
                    short_balance=short_balance,
                )
            )
            inserted += 1

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(rows),
            "inserted": inserted,
        }

    async def sync_margin(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的融資融券（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailyMargin.date).where(
            StockDailyMargin.stock_id == stock_id,
            StockDailyMargin.date >= start_date,
            StockDailyMargin.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockMarginPurchaseShortSale", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            # FinMind 欄位
            margin_buy = self._safe_int(r.get("MarginPurchaseBuy", 0))
            margin_sell = self._safe_int(r.get("MarginPurchaseSell", 0))
            margin_balance = self._safe_int(r.get("MarginPurchaseTodayBalance", 0))
            short_buy = self._safe_int(r.get("ShortSaleBuy", 0))
            short_sell = self._safe_int(r.get("ShortSaleSell", 0))
            short_balance = self._safe_int(r.get("ShortSaleTodayBalance", 0))

            self._session.add(
                StockDailyMargin(
                    stock_id=stock_id,
                    date=r_date,
                    margin_buy=margin_buy,
                    margin_sell=margin_sell,
                    margin_balance=margin_balance,
                    short_buy=short_buy,
                    short_sell=short_sell,
                    short_balance=short_balance,
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailyMargin.date).where(
            StockDailyMargin.stock_id == stock_id,
            StockDailyMargin.date >= start_date,
            StockDailyMargin.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    def get_margin_status(self, start_date: date, end_date: date) -> dict:
        """取得融資融券資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailyMargin.date),
                func.max(StockDailyMargin.date),
                func.count(),
            ).where(
                StockDailyMargin.stock_id == stock.stock_id,
                StockDailyMargin.date >= start_date,
                StockDailyMargin.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 還原股價 (yfinance)
    # =========================================================================

    async def sync_adj(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的還原股價（使用 yfinance）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailyAdj.date).where(
            StockDailyAdj.stock_id == stock_id,
            StockDailyAdj.date >= start_date,
            StockDailyAdj.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 yfinance 取得資料
        ticker = yf.Ticker(f"{stock_id}.TW")
        # 多抓一天以確保範圍正確
        df = ticker.history(
            start=min(missing_dates).isoformat(),
            end=(max(missing_dates) + timedelta(days=1)).isoformat(),
        )

        if df.empty:
            return {"fetched": 0, "inserted": 0, "missing_dates": [d.isoformat() for d in missing_dates]}

        inserted = 0
        for idx, row in df.iterrows():
            r_date = idx.date()
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            adj_close = row.get("Close")  # yfinance 的 Close 已經是還原價
            if adj_close is None or adj_close <= 0:
                continue

            self._session.add(
                StockDailyAdj(
                    stock_id=stock_id,
                    date=r_date,
                    adj_close=Decimal(str(round(adj_close, 2))),
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailyAdj.date).where(
            StockDailyAdj.stock_id == stock_id,
            StockDailyAdj.date >= start_date,
            StockDailyAdj.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(df),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    async def sync_adj_bulk(self, target_date: date) -> dict:
        """
        同步全市場還原股價（用 yfinance 批次）
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]

        if not stock_ids:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 取得已有資料
        stmt = select(StockDailyAdj.stock_id).where(StockDailyAdj.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 批次取得資料（一次最多 50 檔以避免超時）
        inserted = 0
        batch_size = 50

        for i in range(0, len(stock_ids), batch_size):
            batch = stock_ids[i:i + batch_size]
            tickers = [f"{sid}.TW" for sid in batch if sid not in existing_stocks]

            if not tickers:
                continue

            # yfinance 批次下載
            data = yf.download(
                tickers,
                start=target_date.isoformat(),
                end=(target_date + timedelta(days=1)).isoformat(),
                progress=False,
            )

            if data.empty:
                continue

            # 處理單一股票的情況（columns 結構不同）
            if len(tickers) == 1:
                stock_id = tickers[0].replace(".TW", "")
                if stock_id not in existing_stocks:
                    close_val = data["Close"].iloc[0] if not data["Close"].empty else None
                    if close_val and close_val > 0:
                        self._session.add(
                            StockDailyAdj(
                                stock_id=stock_id,
                                date=target_date,
                                adj_close=Decimal(str(round(close_val, 2))),
                            )
                        )
                        inserted += 1
            else:
                # 多股票的情況
                for ticker in tickers:
                    stock_id = ticker.replace(".TW", "")
                    if stock_id in existing_stocks:
                        continue

                    try:
                        close_val = data["Close"][ticker].iloc[0]
                        if close_val and close_val > 0:
                            self._session.add(
                                StockDailyAdj(
                                    stock_id=stock_id,
                                    date=target_date,
                                    adj_close=Decimal(str(round(close_val, 2))),
                                )
                            )
                            inserted += 1
                    except (KeyError, IndexError):
                        continue

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(stock_ids),
            "inserted": inserted,
        }

    def get_adj_status(self, start_date: date, end_date: date) -> dict:
        """取得還原股價資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailyAdj.date),
                func.max(StockDailyAdj.date),
                func.count(),
            ).where(
                StockDailyAdj.stock_id == stock.stock_id,
                StockDailyAdj.date >= start_date,
                StockDailyAdj.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 外資持股
    # =========================================================================

    TWSE_SHAREHOLDING_URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"

    async def sync_shareholding_bulk(self, target_date: date) -> dict:
        """
        同步全市場外資持股（用 TWSE RWD bulk API）
        只儲存股票池內的股票
        Returns: {"date": str, "total": int, "inserted": int}
        """
        # 取得股票池
        stmt = select(StockUniverse.stock_id)
        universe = {row[0] for row in self._session.execute(stmt).fetchall()}

        if not universe:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        # 呼叫 TWSE RWD API（需帶日期參數）
        params = {
            "date": target_date.strftime("%Y%m%d"),
            "selectType": "ALLBUT0999",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.TWSE_SHAREHOLDING_URL, params=params, timeout=30)
            data = resp.json()

        if data.get("stat") != "OK":
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "TWSE API failed"}

        # 取得已有資料
        stmt = select(StockDailyShareholding.stock_id).where(StockDailyShareholding.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}

        rows = data.get("data", [])
        inserted = 0

        for row in rows:
            # MI_QFIIS 格式（12 欄）:
            # 0: 證券代號, 1: 證券名稱, 2: ISIN, 3: 發行股數,
            # 4: 尚可投資股數, 5: 全體外資持有股數, 6: 尚可投資比率,
            # 7: 全體外資持股比率, 8: 外資投資上限比率, 9: 陸資投資上限比率, 10-11: 其他
            if len(row) < 10:
                continue

            stock_id = row[0].strip()

            # 只儲存股票池內的股票
            if stock_id not in universe:
                continue
            # 跳過已存在的
            if stock_id in existing_stocks:
                continue

            self._session.add(
                StockDailyShareholding(
                    stock_id=stock_id,
                    date=target_date,
                    total_shares=self._safe_int(row[3]),                    # 發行股數
                    foreign_shares=self._safe_int(row[5]),                  # 全體外資持有股數
                    foreign_ratio=self._safe_decimal(row[7]) or Decimal("0"),  # 全體外資持股比率
                    foreign_remaining_shares=self._safe_int(row[4]),        # 尚可投資股數
                    foreign_remaining_ratio=self._safe_decimal(row[6]) or Decimal("0"),  # 尚可投資比率
                    foreign_upper_limit_ratio=self._safe_decimal(row[8]) or Decimal("0"),  # 外資投資上限比率
                    chinese_upper_limit_ratio=self._safe_decimal(row[9]) or Decimal("0"),  # 陸資投資上限比率
                )
            )
            inserted += 1

        self._session.commit()

        return {
            "date": target_date.isoformat(),
            "total": len(rows),
            "inserted": inserted,
        }

    async def sync_shareholding(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的外資持股（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailyShareholding.date).where(
            StockDailyShareholding.stock_id == stock_id,
            StockDailyShareholding.date >= start_date,
            StockDailyShareholding.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockShareholding", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            self._session.add(
                StockDailyShareholding(
                    stock_id=stock_id,
                    date=r_date,
                    total_shares=self._safe_int(r.get("NumberOfSharesIssued", 0)),
                    foreign_shares=self._safe_int(r.get("ForeignInvestmentShares", 0)),
                    foreign_ratio=self._safe_decimal(r.get("ForeignInvestmentSharesRatio")) or Decimal("0"),
                    foreign_remaining_shares=self._safe_int(r.get("ForeignInvestmentRemainingShares", 0)),
                    foreign_remaining_ratio=self._safe_decimal(r.get("ForeignInvestmentRemainRatio")) or Decimal("0"),
                    foreign_upper_limit_ratio=self._safe_decimal(r.get("ForeignInvestmentUpperLimitRatio")) or Decimal("0"),
                    chinese_upper_limit_ratio=self._safe_decimal(r.get("ChineseInvestmentUpperLimitRatio")) or Decimal("0"),
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailyShareholding.date).where(
            StockDailyShareholding.stock_id == stock_id,
            StockDailyShareholding.date >= start_date,
            StockDailyShareholding.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    def get_shareholding_status(self, start_date: date, end_date: date) -> dict:
        """取得外資持股資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailyShareholding.date),
                func.max(StockDailyShareholding.date),
                func.count(),
            ).where(
                StockDailyShareholding.stock_id == stock.stock_id,
                StockDailyShareholding.date >= start_date,
                StockDailyShareholding.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 借券明細
    # =========================================================================

    async def sync_securities_lending(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步單一股票的借券明細（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_dates": list}
        """
        # 取得交易日
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 取得已有資料的日期
        stmt = select(StockDailySecuritiesLending.date).where(
            StockDailySecuritiesLending.stock_id == stock_id,
            StockDailySecuritiesLending.date >= start_date,
            StockDailySecuritiesLending.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        # 計算缺少的日期
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockSecuritiesLending", {
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            r_date = date.fromisoformat(r["date"])
            if r_date not in missing_dates:
                continue
            if r_date in existing_dates:
                continue

            self._session.add(
                StockDailySecuritiesLending(
                    stock_id=stock_id,
                    date=r_date,
                    lending_volume=self._safe_int(r.get("Volume", 0)),
                    lending_balance=self._safe_int(r.get("balance", 0)),
                )
            )
            inserted += 1

        self._session.commit()

        # 重新計算還缺少的日期
        stmt = select(StockDailySecuritiesLending.date).where(
            StockDailySecuritiesLending.stock_id == stock_id,
            StockDailySecuritiesLending.date >= start_date,
            StockDailySecuritiesLending.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_dates": [d.isoformat() for d in still_missing],
        }

    def get_securities_lending_status(self, start_date: date, end_date: date) -> dict:
        """取得借券明細資料狀態"""
        from sqlalchemy import func

        # 取得整體交易日數（用於顯示）
        trading_days = self.count_trading_days(start_date, end_date)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            stmt = select(
                func.min(StockDailySecuritiesLending.date),
                func.max(StockDailySecuritiesLending.date),
                func.count(),
            ).where(
                StockDailySecuritiesLending.stock_id == stock.stock_id,
                StockDailySecuritiesLending.date >= start_date,
                StockDailySecuritiesLending.date <= end_date,
            )
            result = self._session.execute(stmt).fetchone()
            earliest, latest, total = result

            # 用該股票的首筆資料日期計算覆蓋率（考慮最低筆數門檻）
            if earliest:
                expected_days = self.count_trading_days(earliest, end_date)
                missing = max(0, expected_days - total)
                coverage = self._calc_coverage(total, expected_days, self.MIN_DAILY_RECORDS)
            else:
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest.isoformat() if earliest else None,
                "latest_date": latest.isoformat() if latest else None,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "trading_days": trading_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stocks": stocks,
        }

    # =========================================================================
    # 月營收（低頻）
    # =========================================================================

    @staticmethod
    def _is_revenue_available(year: int, month: int) -> bool:
        """判斷某月營收是否應已公布（次月10日）"""
        today = date.today()
        if month == 12:
            deadline = date(year + 1, 1, 10)
        else:
            deadline = date(year, month + 1, 10)
        return today >= deadline

    @staticmethod
    def _get_expected_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
        """取得應有的月份列表（考慮公布時程）"""
        today = date.today()
        months = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            # 檢查該月營收是否應已公布
            if month == 12:
                deadline = date(year + 1, 1, 10)
            else:
                deadline = date(year, month + 1, 10)

            if today >= deadline:
                months.append((year, month))

            # 下個月
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return months

    async def sync_monthly_revenue(self, stock_id: str, start_year: int, end_year: int) -> dict:
        """
        同步單一股票的月營收（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_months": list}
        """
        # 計算應有的月份
        expected_months = self._get_expected_months(start_year, 1, end_year, 12)
        if not expected_months:
            return {"fetched": 0, "inserted": 0, "missing_months": []}

        # 取得已有資料的月份
        start_val = start_year * 100 + 1
        end_val = end_year * 100 + 12
        stmt = select(StockMonthlyRevenue.year, StockMonthlyRevenue.month).where(
            StockMonthlyRevenue.stock_id == stock_id,
            (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) >= start_val,
            (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) <= end_val,
        )
        existing_months = {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

        # 計算缺少的月份
        missing_months = [m for m in expected_months if m not in existing_months]
        if not missing_months:
            return {"fetched": 0, "inserted": 0, "missing_months": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockMonthRevenue", {
            "data_id": stock_id,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
        })

        records = data.get("data", [])
        inserted = 0

        for r in records:
            # FinMind 用 date 欄位表示月份（格式: 2024-01-01 表示 2024年1月）
            r_date = date.fromisoformat(r["date"])
            r_year, r_month = r_date.year, r_date.month

            if (r_year, r_month) in existing_months:
                continue
            if (r_year, r_month) not in expected_months:
                continue

            revenue = self._safe_decimal(r.get("revenue"))
            if revenue is None:
                continue

            self._session.add(
                StockMonthlyRevenue(
                    stock_id=stock_id,
                    year=r_year,
                    month=r_month,
                    revenue=revenue,
                    revenue_yoy=self._safe_decimal(r.get("revenue_year_growth_rate")),
                    revenue_mom=self._safe_decimal(r.get("revenue_month_growth_rate")),
                )
            )
            inserted += 1
            existing_months.add((r_year, r_month))

        self._session.commit()

        # 重新計算還缺少的月份
        still_missing = [m for m in expected_months if m not in existing_months]

        return {
            "fetched": len(records),
            "inserted": inserted,
            "missing_months": [f"{y}-{m:02d}" for y, m in still_missing],
        }

    def get_monthly_revenue_status(self, start_year: int, end_year: int) -> dict:
        """取得月營收資料狀態"""
        from sqlalchemy import func

        # 計算應有的月份數
        expected_months = self._get_expected_months(start_year, 1, end_year, 12)

        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 統計該股票的資料
            start_val = start_year * 100 + 1
            end_val = end_year * 100 + 12
            stmt = select(
                func.min(StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month),
                func.max(StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month),
                func.count(),
            ).where(
                StockMonthlyRevenue.stock_id == stock.stock_id,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) >= start_val,
                (StockMonthlyRevenue.year * 100 + StockMonthlyRevenue.month) <= end_val,
            )
            result = self._session.execute(stmt).fetchone()
            earliest_val, latest_val, total = result

            if earliest_val:
                earliest_year = earliest_val // 100
                earliest_month = earliest_val % 100
                latest_year = latest_val // 100
                latest_month = latest_val % 100
                earliest_str = f"{earliest_year}-{earliest_month:02d}"
                latest_str = f"{latest_year}-{latest_month:02d}"

                # 從首筆資料起算的期望月份數（考慮最低筆數門檻）
                months_from_start = self._get_expected_months(earliest_year, earliest_month, end_year, 12)
                expected = len(months_from_start)
                missing = max(0, expected - total)
                coverage = self._calc_coverage(total, expected, self.MIN_MONTHLY_RECORDS)
            else:
                earliest_str = None
                latest_str = None
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_month": earliest_str,
                "latest_month": latest_str,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "expected_months": len(expected_months),
            "start_year": start_year,
            "end_year": end_year,
            "stocks": stocks,
        }

    # =========================================================================
    # 季度財報（低頻）
    # =========================================================================

    # 財報公布期限：Q1: 5/15, Q2: 8/14, Q3: 11/14, Q4: 隔年 3/31
    QUARTER_DEADLINES = {
        1: (5, 15),
        2: (8, 14),
        3: (11, 14),
        4: (3, 31),
    }

    @classmethod
    def _is_quarter_available(cls, year: int, quarter: int) -> bool:
        """判斷某季財報是否應已公布"""
        today = date.today()
        month, day = cls.QUARTER_DEADLINES[quarter]
        deadline_year = year + 1 if quarter == 4 else year
        return today >= date(deadline_year, month, day)

    @classmethod
    def _get_expected_quarters(cls, start_year: int, end_year: int) -> list[tuple[int, int]]:
        """取得應有的季度列表（考慮公布時程）"""
        quarters = []
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                if cls._is_quarter_available(year, q):
                    quarters.append((year, q))
        return quarters

    async def sync_quarterly_financial(self, stock_id: str, start_year: int, end_year: int) -> dict:
        """
        同步單一股票的季度財報（綜合損益表，使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_quarters": list}
        """
        expected_quarters = self._get_expected_quarters(start_year, end_year)
        if not expected_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        # 取得已有資料的季度
        start_val = start_year * 10 + 1
        end_val = end_year * 10 + 4
        stmt = select(StockQuarterlyFinancial.year, StockQuarterlyFinancial.quarter).where(
            StockQuarterlyFinancial.stock_id == stock_id,
            (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) >= start_val,
            (StockQuarterlyFinancial.year * 10 + StockQuarterlyFinancial.quarter) <= end_val,
        )
        existing_quarters = {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

        missing_quarters = [q for q in expected_quarters if q not in existing_quarters]
        if not missing_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        # 用 FinMind 補缺少的資料
        data = await self._fetch_finmind("TaiwanStockFinancialStatements", {
            "data_id": stock_id,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
        })

        records = data.get("data", [])
        # FinMind 返回多列，需要按 (stock_id, date) 分組
        grouped: dict[tuple, dict] = {}
        for r in records:
            key = (r["stock_id"], r["date"])
            if key not in grouped:
                grouped[key] = {"stock_id": r["stock_id"], "date": r["date"]}
            grouped[key][r["type"]] = r["value"]

        inserted = 0
        for key, values in grouped.items():
            r_date = date.fromisoformat(values["date"])
            r_year = r_date.year
            r_quarter = (r_date.month - 1) // 3 + 1

            if (r_year, r_quarter) in existing_quarters:
                continue
            if (r_year, r_quarter) not in expected_quarters:
                continue

            self._session.add(
                StockQuarterlyFinancial(
                    stock_id=stock_id,
                    year=r_year,
                    quarter=r_quarter,
                    revenue=self._safe_decimal(values.get("Revenue")),
                    gross_profit=self._safe_decimal(values.get("GrossProfit")),
                    operating_income=self._safe_decimal(values.get("OperatingIncome")),
                    net_income=self._safe_decimal(values.get("NetIncome")),
                    eps=self._safe_decimal(values.get("EPS")),
                )
            )
            inserted += 1
            existing_quarters.add((r_year, r_quarter))

        self._session.commit()

        still_missing = [q for q in expected_quarters if q not in existing_quarters]
        return {
            "fetched": len(grouped),
            "inserted": inserted,
            "missing_quarters": [f"{y}Q{q}" for y, q in still_missing],
        }

    async def sync_quarterly_balance(self, stock_id: str, start_year: int, end_year: int) -> dict:
        """
        同步單一股票的資產負債表（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_quarters": list}
        """
        expected_quarters = self._get_expected_quarters(start_year, end_year)
        if not expected_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        start_val = start_year * 10 + 1
        end_val = end_year * 10 + 4
        stmt = select(StockQuarterlyBalance.year, StockQuarterlyBalance.quarter).where(
            StockQuarterlyBalance.stock_id == stock_id,
            (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) >= start_val,
            (StockQuarterlyBalance.year * 10 + StockQuarterlyBalance.quarter) <= end_val,
        )
        existing_quarters = {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

        missing_quarters = [q for q in expected_quarters if q not in existing_quarters]
        if not missing_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        data = await self._fetch_finmind("TaiwanStockBalanceSheet", {
            "data_id": stock_id,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
        })

        records = data.get("data", [])
        grouped: dict[tuple, dict] = {}
        for r in records:
            key = (r["stock_id"], r["date"])
            if key not in grouped:
                grouped[key] = {"stock_id": r["stock_id"], "date": r["date"]}
            grouped[key][r["type"]] = r["value"]

        inserted = 0
        for key, values in grouped.items():
            r_date = date.fromisoformat(values["date"])
            r_year = r_date.year
            r_quarter = (r_date.month - 1) // 3 + 1

            if (r_year, r_quarter) in existing_quarters:
                continue
            if (r_year, r_quarter) not in expected_quarters:
                continue

            self._session.add(
                StockQuarterlyBalance(
                    stock_id=stock_id,
                    year=r_year,
                    quarter=r_quarter,
                    total_assets=self._safe_decimal(values.get("TotalAssets")),
                    total_liabilities=self._safe_decimal(values.get("TotalLiabilities")),
                    total_equity=self._safe_decimal(values.get("TotalEquity")),
                )
            )
            inserted += 1
            existing_quarters.add((r_year, r_quarter))

        self._session.commit()

        still_missing = [q for q in expected_quarters if q not in existing_quarters]
        return {
            "fetched": len(grouped),
            "inserted": inserted,
            "missing_quarters": [f"{y}Q{q}" for y, q in still_missing],
        }

    async def sync_quarterly_cashflow(self, stock_id: str, start_year: int, end_year: int) -> dict:
        """
        同步單一股票的現金流量表（使用 FinMind）
        Returns: {"fetched": int, "inserted": int, "missing_quarters": list}
        """
        expected_quarters = self._get_expected_quarters(start_year, end_year)
        if not expected_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        start_val = start_year * 10 + 1
        end_val = end_year * 10 + 4
        stmt = select(StockQuarterlyCashFlow.year, StockQuarterlyCashFlow.quarter).where(
            StockQuarterlyCashFlow.stock_id == stock_id,
            (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) >= start_val,
            (StockQuarterlyCashFlow.year * 10 + StockQuarterlyCashFlow.quarter) <= end_val,
        )
        existing_quarters = {(r[0], r[1]) for r in self._session.execute(stmt).fetchall()}

        missing_quarters = [q for q in expected_quarters if q not in existing_quarters]
        if not missing_quarters:
            return {"fetched": 0, "inserted": 0, "missing_quarters": []}

        data = await self._fetch_finmind("TaiwanStockCashFlowsStatement", {
            "data_id": stock_id,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
        })

        records = data.get("data", [])
        grouped: dict[tuple, dict] = {}
        for r in records:
            key = (r["stock_id"], r["date"])
            if key not in grouped:
                grouped[key] = {"stock_id": r["stock_id"], "date": r["date"]}
            grouped[key][r["type"]] = r["value"]

        inserted = 0
        for key, values in grouped.items():
            r_date = date.fromisoformat(values["date"])
            r_year = r_date.year
            r_quarter = (r_date.month - 1) // 3 + 1

            if (r_year, r_quarter) in existing_quarters:
                continue
            if (r_year, r_quarter) not in expected_quarters:
                continue

            operating = self._safe_decimal(values.get("OperatingActivities"))
            investing = self._safe_decimal(values.get("InvestingActivities"))
            financing = self._safe_decimal(values.get("FinancingActivities"))
            free_cf = None
            if operating is not None and investing is not None:
                free_cf = operating + investing

            self._session.add(
                StockQuarterlyCashFlow(
                    stock_id=stock_id,
                    year=r_year,
                    quarter=r_quarter,
                    operating_cf=operating,
                    investing_cf=investing,
                    financing_cf=financing,
                    free_cf=free_cf,
                )
            )
            inserted += 1
            existing_quarters.add((r_year, r_quarter))

        self._session.commit()

        still_missing = [q for q in expected_quarters if q not in existing_quarters]
        return {
            "fetched": len(grouped),
            "inserted": inserted,
            "missing_quarters": [f"{y}Q{q}" for y, q in still_missing],
        }

    def _get_quarterly_status(self, model_class, start_year: int, end_year: int) -> dict:
        """取得季度資料狀態（共用方法）"""
        from sqlalchemy import func

        expected_quarters = self._get_expected_quarters(start_year, end_year)

        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            start_val = start_year * 10 + 1
            end_val = end_year * 10 + 4
            stmt = select(
                func.min(model_class.year * 10 + model_class.quarter),
                func.max(model_class.year * 10 + model_class.quarter),
                func.count(),
            ).where(
                model_class.stock_id == stock.stock_id,
                (model_class.year * 10 + model_class.quarter) >= start_val,
                (model_class.year * 10 + model_class.quarter) <= end_val,
            )
            result = self._session.execute(stmt).fetchone()
            earliest_val, latest_val, total = result

            if earliest_val:
                earliest_year = earliest_val // 10
                earliest_quarter = earliest_val % 10
                latest_year = latest_val // 10
                latest_quarter = latest_val % 10
                earliest_str = f"{earliest_year}Q{earliest_quarter}"
                latest_str = f"{latest_year}Q{latest_quarter}"

                # 考慮最低筆數門檻
                quarters_from_start = [q for q in expected_quarters if q >= (earliest_year, earliest_quarter)]
                expected = len(quarters_from_start)
                missing = max(0, expected - total)
                coverage = self._calc_coverage(total, expected, self.MIN_QUARTERLY_RECORDS)
            else:
                earliest_str = None
                latest_str = None
                missing = 0
                coverage = 0

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_quarter": earliest_str,
                "latest_quarter": latest_str,
                "total_records": total,
                "missing_count": missing,
                "coverage_pct": round(coverage, 1),
            })

        return {
            "expected_quarters": len(expected_quarters),
            "start_year": start_year,
            "end_year": end_year,
            "stocks": stocks,
        }

    def get_quarterly_financial_status(self, start_year: int, end_year: int) -> dict:
        """取得綜合損益表資料狀態"""
        return self._get_quarterly_status(StockQuarterlyFinancial, start_year, end_year)

    def get_quarterly_balance_status(self, start_year: int, end_year: int) -> dict:
        """取得資產負債表資料狀態"""
        return self._get_quarterly_status(StockQuarterlyBalance, start_year, end_year)

    def get_quarterly_cashflow_status(self, start_year: int, end_year: int) -> dict:
        """取得現金流量表資料狀態"""
        return self._get_quarterly_status(StockQuarterlyCashFlow, start_year, end_year)

    # =========================================================================
    # 股利政策 (Dividend)
    # =========================================================================

    async def sync_dividend(self, stock_id: str, start_date: date, end_date: date) -> dict:
        """
        同步個股股利資料（使用 FinMind）
        Returns: {"fetched": int, "inserted": int}
        """
        data = await self._fetch_finmind("TaiwanStockDividend", {
            "data_id": stock_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }, timeout=30)

        rows = data.get("data", [])

        # 取得已存在的除息日
        stmt = select(StockDividend.ex_date).where(
            StockDividend.stock_id == stock_id,
            StockDividend.ex_date >= start_date,
            StockDividend.ex_date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}

        inserted = 0
        for row in rows:
            ex_date_str = row.get("date")
            if not ex_date_str:
                continue

            ex_date = date.fromisoformat(ex_date_str)

            if ex_date in existing_dates:
                continue

            self._session.add(StockDividend(
                stock_id=stock_id,
                ex_date=ex_date,
                cash_dividend=self._safe_decimal(row.get("CashEarningsDistribution")),
                stock_dividend=self._safe_decimal(row.get("StockEarningsDistribution")),
            ))
            inserted += 1

        self._session.commit()
        return {"fetched": len(rows), "inserted": inserted}

    async def sync_dividend_all(self, start_date: date, end_date: date) -> dict:
        """
        同步全股票池股利資料
        Returns: {"total_stocks": int, "total_fetched": int, "total_inserted": int, "stocks": list}
        """
        # 取得股票池
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        results = []
        total_fetched = 0
        total_inserted = 0

        for stock in universe:
            try:
                result = await self.sync_dividend(stock.stock_id, start_date, end_date)
                results.append({
                    "stock_id": stock.stock_id,
                    "name": stock.name,
                    "fetched": result["fetched"],
                    "inserted": result["inserted"],
                    "status": "success",
                })
                total_fetched += result["fetched"]
                total_inserted += result["inserted"]
            except Exception as e:
                results.append({
                    "stock_id": stock.stock_id,
                    "name": stock.name,
                    "fetched": 0,
                    "inserted": 0,
                    "status": "error",
                    "error": str(e),
                })

        return {
            "total_stocks": len(universe),
            "total_fetched": total_fetched,
            "total_inserted": total_inserted,
            "stocks": results,
        }

    def get_dividend_status(self, start_year: int, end_year: int) -> dict:
        """
        取得股利資料狀態
        Returns: {"start_year": int, "end_year": int, "stocks": list}
        """
        from sqlalchemy import func, extract

        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        universe = self._session.execute(stmt).scalars().all()

        stocks = []
        for stock in universe:
            # 取得資料範圍和記錄數
            stmt = select(
                func.min(StockDividend.ex_date),
                func.max(StockDividend.ex_date),
                func.count(),
            ).where(
                StockDividend.stock_id == stock.stock_id,
                extract("year", StockDividend.ex_date) >= start_year,
                extract("year", StockDividend.ex_date) <= end_year,
            )
            result = self._session.execute(stmt).fetchone()
            earliest_date, latest_date, total = result

            # 取得有資料的年份
            stmt = select(
                extract("year", StockDividend.ex_date).label("year")
            ).where(
                StockDividend.stock_id == stock.stock_id,
                extract("year", StockDividend.ex_date) >= start_year,
                extract("year", StockDividend.ex_date) <= end_year,
            ).distinct()
            years_with_data = [int(row[0]) for row in self._session.execute(stmt).fetchall()]

            # 計算缺少的年份
            all_years = set(range(start_year, end_year + 1))
            missing_years = sorted(all_years - set(years_with_data))

            stocks.append({
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": earliest_date.isoformat() if earliest_date else None,
                "latest_date": latest_date.isoformat() if latest_date else None,
                "total_records": total,
                "years_with_data": sorted(years_with_data),
                "missing_years": missing_years,
            })

        return {
            "start_year": start_year,
            "end_year": end_year,
            "stocks": stocks,
        }

    # =========================================================================
    # 工具方法
    # =========================================================================

    @staticmethod
    def _safe_decimal(value) -> Decimal | None:
        if value is None or value == "" or value in ("--", "-"):
            return None
        try:
            cleaned = str(value).replace(",", "")
            return Decimal(cleaned)
        except Exception:
            return None

    @staticmethod
    def _safe_int(value) -> int:
        if value is None or value == "" or value in ("--", "-"):
            return 0
        try:
            cleaned = str(value).replace(",", "")
            return int(cleaned)
        except Exception:
            return 0
