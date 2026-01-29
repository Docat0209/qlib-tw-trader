"""
API 測試共用 Fixtures
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.interfaces.dependencies import get_db
from src.interfaces.exceptions import register_exception_handlers
from src.interfaces.routers import system
from src.repositories.database import Base


@pytest.fixture
def test_engine():
    """建立測試用記憶體資料庫引擎"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from src.repositories import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_db(test_engine):
    """建立測試用 Session"""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(test_db):
    """建立測試用 HTTP Client"""
    app = FastAPI()
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    register_exception_handlers(app)

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c
