"""
股票池 API
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.repositories.models import StockUniverse

router = APIRouter()


class StockInfo(BaseModel):
    """股票資訊"""
    stock_id: str
    name: str
    market_cap: int
    rank: int


class UniverseResponse(BaseModel):
    """股票池回應"""
    name: str
    description: str
    total: int
    stocks: list[StockInfo]
    updated_at: datetime | None


class UniverseStats(BaseModel):
    """股票池統計"""
    total: int
    min_market_cap: int
    max_market_cap: int
    updated_at: datetime | None


@router.get("", response_model=UniverseResponse)
async def get_universe(session: Session = Depends(get_db)):
    """取得股票池"""
    stmt = select(StockUniverse).order_by(StockUniverse.rank)
    result = session.execute(stmt)
    stocks = result.scalars().all()

    updated_at = stocks[0].updated_at if stocks else None

    return UniverseResponse(
        name="tw100",
        description="台股市值前100（排除ETF、KY股）",
        total=len(stocks),
        stocks=[
            StockInfo(
                stock_id=s.stock_id,
                name=s.name,
                market_cap=s.market_cap,
                rank=s.rank,
            )
            for s in stocks
        ],
        updated_at=updated_at,
    )


@router.get("/stats", response_model=UniverseStats)
async def get_universe_stats(session: Session = Depends(get_db)):
    """取得股票池統計"""
    stmt = select(StockUniverse).order_by(StockUniverse.rank)
    result = session.execute(stmt)
    stocks = result.scalars().all()

    if not stocks:
        return UniverseStats(
            total=0,
            min_market_cap=0,
            max_market_cap=0,
            updated_at=None,
        )

    return UniverseStats(
        total=len(stocks),
        min_market_cap=min(s.market_cap for s in stocks),
        max_market_cap=max(s.market_cap for s in stocks),
        updated_at=stocks[0].updated_at,
    )


@router.get("/ids")
async def get_stock_ids(session: Session = Depends(get_db)):
    """取得股票代碼清單"""
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    result = session.execute(stmt)
    stock_ids = [row[0] for row in result.fetchall()]
    return {"stock_ids": stock_ids, "total": len(stock_ids)}


@router.post("/sync")
async def sync_universe(session: Session = Depends(get_db)):
    """從 TWSE 更新股票池"""
    import httpx

    # 取得成交資料
    price_url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
    share_url = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"

    async with httpx.AsyncClient() as client:
        price_resp = await client.get(price_url, timeout=30)
        share_resp = await client.get(share_url, params={"selectType": "ALLBUT0999"}, timeout=30)

    price_data = price_resp.json()
    share_data = share_resp.json()

    if price_data.get("stat") != "OK" or share_data.get("stat") != "OK":
        return {"success": False, "error": "Failed to fetch data from TWSE"}

    # 建立價格字典
    price_map = {}
    for row in price_data.get("data", []):
        if len(row) < 8:
            continue
        stock_id = row[0].strip()
        name = row[1].strip()
        try:
            close = float(row[7].replace(",", "")) if row[7] not in ("--", "-", "") else 0
            price_map[stock_id] = {"close": close, "name": name}
        except:
            continue

    # 建立股本字典
    shares_map = {}
    for row in share_data.get("data", []):
        if len(row) < 4:
            continue
        stock_id = row[0].strip()
        try:
            issued_shares = int(row[3].replace(",", ""))
            shares_map[stock_id] = issued_shares
        except:
            continue

    # 計算市值並篩選
    stocks = []
    for stock_id, price_info in price_map.items():
        name = price_info["name"]

        # 排除條件
        if not stock_id.isdigit() or len(stock_id) != 4:
            continue
        if stock_id.startswith("0"):
            continue
        if "-KY" in name or "KY" in name:
            continue
        if "*" in name or "-創" in name:
            continue
        if price_info["close"] <= 0:
            continue
        if stock_id not in shares_map:
            continue

        market_cap = price_info["close"] * shares_map[stock_id]
        stocks.append({
            "stock_id": stock_id,
            "name": name,
            "market_cap": round(market_cap / 1e8),
        })

    # 按市值排序，取前 100
    stocks.sort(key=lambda x: x["market_cap"], reverse=True)
    top100 = stocks[:100]

    # 清除舊資料
    session.execute(StockUniverse.__table__.delete())

    # 寫入新資料
    now = datetime.now()
    for rank, s in enumerate(top100, 1):
        session.add(StockUniverse(
            stock_id=s["stock_id"],
            name=s["name"],
            market_cap=s["market_cap"],
            rank=rank,
            updated_at=now,
        ))

    session.commit()

    return {
        "success": True,
        "total": len(top100),
        "updated_at": now.isoformat(),
    }
