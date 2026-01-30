"""
Datasets 測試 API
用於列出和測試各種資料集
"""

from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel
import httpx

router = APIRouter()

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


class DatasetInfo(BaseModel):
    """資料集資訊"""
    name: str
    display_name: str
    category: str
    source: str
    status: Literal["available", "needs_accumulation", "not_implemented", "pending"]
    description: str | None = None
    requires_stock_id: bool = True


class DatasetListResponse(BaseModel):
    """資料集列表回應"""
    datasets: list[DatasetInfo]
    total: int


class TestResult(BaseModel):
    """測試結果"""
    dataset: str
    success: bool
    record_count: int
    sample_data: list[dict] | None = None
    error: str | None = None


# 完整的 datasets 定義
ALL_DATASETS = [
    # 技術面
    DatasetInfo(name="TaiwanStockPrice", display_name="日K線", category="technical", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockPriceAdj", display_name="還原股價", category="technical", source="yfinance", status="available"),
    DatasetInfo(name="TaiwanStockPER", display_name="PER/PBR/殖利率", category="technical", source="twse/finmind", status="available"),

    # 籌碼面
    DatasetInfo(name="TaiwanStockMarginPurchaseShortSale", display_name="個股融資融券", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockInstitutionalInvestorsBuySell", display_name="個股三大法人", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockShareholding", display_name="外資持股", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockSecuritiesLending", display_name="借券明細", category="chips", source="twse/finmind", status="available"),

    # 基本面
    DatasetInfo(name="TaiwanStockCashFlowsStatement", display_name="現金流量表", category="fundamental", source="finmind", status="available"),
    DatasetInfo(name="TaiwanStockFinancialStatements", display_name="綜合損益表", category="fundamental", source="finmind", status="available"),
    DatasetInfo(name="TaiwanStockBalanceSheet", display_name="資產負債表", category="fundamental", source="finmind", status="available"),
    DatasetInfo(name="TaiwanStockDividend", display_name="股利政策", category="fundamental", source="finmind", status="available"),
    DatasetInfo(name="TaiwanStockMonthRevenue", display_name="月營收", category="fundamental", source="finmind", status="available"),

    # 待定 - 需評估或累積資料
    DatasetInfo(name="TaiwanStockDayTrading", display_name="當沖成交量值", category="technical", source="finmind", status="pending", description="非核心因子，資料完整度低"),
    DatasetInfo(name="TaiwanStockHoldingSharesPer", display_name="股權分級表", category="chips", source="twse", status="pending", description="籌碼集中度，需累積"),
    DatasetInfo(name="TaiwanStockTradingDailyReport", display_name="分點資料", category="chips", source="twse", status="pending", description="主力券商進出，需累積"),
    DatasetInfo(name="TaiwanFuturesDaily", display_name="期貨日成交", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股期貨對應"),
    DatasetInfo(name="TaiwanOptionDaily", display_name="選擇權日成交", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股選擇權對應"),
    DatasetInfo(name="TaiwanFuturesInstitutionalInvestors", display_name="期貨三大法人", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股期貨對應"),
]


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    category: str | None = Query(None, description="篩選類別"),
    status: str | None = Query(None, description="篩選狀態"),
):
    """列出所有資料集"""
    datasets = ALL_DATASETS

    if category:
        datasets = [d for d in datasets if d.category == category]
    if status:
        datasets = [d for d in datasets if d.status == status]

    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/test/{dataset_name}", response_model=TestResult)
async def test_dataset(
    dataset_name: str,
    stock_id: str = Query("2330", description="股票代碼"),
    days: int = Query(5, description="測試天數", ge=1, le=30),
):
    """測試單一資料集"""
    # 找到資料集定義
    dataset = next((d for d in ALL_DATASETS if d.name == dataset_name), None)
    if not dataset:
        return TestResult(
            dataset=dataset_name,
            success=False,
            record_count=0,
            error=f"Dataset not found: {dataset_name}",
        )

    if dataset.status == "not_implemented":
        return TestResult(
            dataset=dataset_name,
            success=False,
            record_count=0,
            error="Dataset not implemented yet",
        )

    # 準備日期範圍
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    # 根據資料集類型呼叫不同的 API
    try:
        params = {
            "dataset": dataset_name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        # 需要 stock_id 的資料集
        if dataset.requires_stock_id:
            params["data_id"] = stock_id
        else:
            # 某些資料集需要特定的 data_id
            if dataset_name == "TaiwanFuturesDaily":
                params["data_id"] = "TX"
            elif dataset_name == "TaiwanOptionDaily":
                params["data_id"] = "TXO"

        async with httpx.AsyncClient() as client:
            resp = await client.get(FINMIND_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 200:
            return TestResult(
                dataset=dataset_name,
                success=False,
                record_count=0,
                error=data.get("msg", "Unknown error"),
            )

        records = data.get("data", [])
        return TestResult(
            dataset=dataset_name,
            success=True,
            record_count=len(records),
            sample_data=records[:3] if records else None,
        )

    except Exception as e:
        return TestResult(
            dataset=dataset_name,
            success=False,
            record_count=0,
            error=str(e),
        )


@router.get("/categories")
async def list_categories():
    """列出所有類別"""
    categories = {}
    for d in ALL_DATASETS:
        if d.category not in categories:
            categories[d.category] = {"name": d.category, "count": 0, "available": 0}
        categories[d.category]["count"] += 1
        if d.status == "available":
            categories[d.category]["available"] += 1

    return {
        "categories": [
            {"id": k, "name": _category_name(k), "total": v["count"], "available": v["available"]}
            for k, v in categories.items()
        ]
    }


def _category_name(cat: str) -> str:
    """類別英轉中"""
    return {
        "technical": "技術面",
        "chips": "籌碼面",
        "fundamental": "基本面",
        "derivatives": "衍生品",
        "macro": "總經指標",
    }.get(cat, cat)
