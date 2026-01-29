"""
FastAPI 依賴注入
"""

from typing import Generator

from sqlalchemy.orm import Session

from src.repositories.database import get_session as db_get_session
from src.services.data_service import DataService


def get_db() -> Generator[Session, None, None]:
    """取得資料庫 Session（請求結束後自動關閉）"""
    session = db_get_session()
    try:
        yield session
    finally:
        session.close()


def get_data_service(session: Session) -> DataService:
    """取得 DataService 實例"""
    return DataService(session)
