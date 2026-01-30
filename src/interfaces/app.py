"""
FastAPI 應用程式
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.interfaces.exceptions import register_exception_handlers
from src.interfaces.routers import backtest, dashboard, datasets, factor, model, performance, portfolio, sync, system, universe, websocket
from src.repositories.database import init_db


def create_app() -> FastAPI:
    """建立 FastAPI 應用程式"""
    app = FastAPI(
        title="qlib-tw-trader API",
        description="台灣股票交易與預測系統 API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 設定（允許前端存取）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 註冊路由
    app.include_router(sync.router, prefix="/api/v1/sync", tags=["sync"])
    app.include_router(universe.router, prefix="/api/v1/universe", tags=["universe"])
    app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    app.include_router(factor.router, prefix="/api/v1/factors", tags=["factors"])
    app.include_router(model.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
    app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"])

    # 註冊例外處理
    register_exception_handlers(app)

    # 初始化資料庫
    init_db()

    return app


app = create_app()
