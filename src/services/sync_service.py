"""
資料同步服務
"""

from datetime import date, timedelta
from decimal import Decimal

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import StockDaily, StockDailyInstitutional, StockDailyPER, StockUniverse, TradingCalendar


class SyncService:
    """資料同步服務"""

    FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
    TWSE_RWD_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
    TWSE_PER_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_ALL"
    TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"

    def __init__(self, session: Session):
        self._session = session

    # =========================================================================
    # 交易日曆
    # =========================================================================

    async def sync_trading_calendar(self, start_date: date, end_date: date) -> int:
        """
        同步交易日曆（用 0050 ETF 推算交易日）
        Returns: 新增的交易日數量
        """
        # 從 FinMind 取得 0050 的日K資料
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": "0050",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.FINMIND_URL, params=params, timeout=60)
            data = resp.json()

        if data.get("status") != 200:
            raise RuntimeError(f"FinMind API error: {data.get('msg')}")

        records = data.get("data", [])
        trading_dates = {date.fromisoformat(r["date"]) for r in records}

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
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.FINMIND_URL, params=params, timeout=60)
            data = resp.json()

        if data.get("status") != 200:
            raise RuntimeError(f"FinMind API error: {data.get('msg')}")

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
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.FINMIND_URL, params=params, timeout=60)
            data = resp.json()

        if data.get("status") != 200:
            raise RuntimeError(f"FinMind API error: {data.get('msg')}")

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

        # 取得交易日數
        stmt = select(func.count()).select_from(TradingCalendar).where(
            TradingCalendar.date >= start_date,
            TradingCalendar.date <= end_date,
            TradingCalendar.is_trading_day == True,
        )
        trading_days = self._session.execute(stmt).scalar() or 0

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

            missing = max(0, trading_days - total) if trading_days > 0 else 0
            coverage = (total / trading_days * 100) if trading_days > 0 else 0

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
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": min(missing_dates).isoformat(),
            "end_date": max(missing_dates).isoformat(),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.FINMIND_URL, params=params, timeout=60)
            data = resp.json()

        if data.get("status") != 200:
            raise RuntimeError(f"FinMind API error: {data.get('msg')}")

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

        # 取得交易日數
        stmt = select(func.count()).select_from(TradingCalendar).where(
            TradingCalendar.date >= start_date,
            TradingCalendar.date <= end_date,
            TradingCalendar.is_trading_day == True,
        )
        trading_days = self._session.execute(stmt).scalar() or 0

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

            missing = max(0, trading_days - total) if trading_days > 0 else 0
            coverage = (total / trading_days * 100) if trading_days > 0 else 0

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
